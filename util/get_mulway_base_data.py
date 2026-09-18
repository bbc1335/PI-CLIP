import cv2
import numpy as np
import argparse
import os.path as osp
from tqdm import tqdm
# from .util import get_train_val_set, check_makedirs
from util import get_train_val_set, check_makedirs

# Get the annotations of base categories

# root_path
# ├── BAM/
# │   ├── util/
# │   ├── config/
# │   ├── model/
# │   ├── README.md
# │   ├── train.py
# │   ├── train_base.py
# │   └── test.py
# └── data/
#     ├── base_annotation/   # the scripts to create THIS folder
#     │   ├── pascal/
#     │   │   ├── train/   
#     │   │   │   ├── 0/     # annotations of PASCAL-5^0
#     │   │   │   ├── 1/
#     │   │   │   ├── 2/
#     │   │   │   └── 3/
#     │   │   └── val/      
#     │   └── coco/          # the same file structure for COCO
#     ├── VOCdevkit2012/
#     └── MSCOCO2014/

parser = argparse.ArgumentParser()
parser.add_argument('--data-set', choices=['pascal', 'coco'], default='pascal')
parser.add_argument(
    '--data-root',
    default='/home/bbc1335/Documents/Dataset/VOCdevkit/VOC2012',
    help='Dataset root containing the original images and annotations.',
)
parser.add_argument(
    '--list-root',
    default=None,
    help=(
        'Directory containing the train/val fss lists. If omitted, '
        'lists/<data-set>/fss_list is used.'
    ),
)
args = parser.parse_args()

args.use_split_coco = True
project_root = osp.dirname(osp.dirname(osp.abspath(__file__)))

for sp in [0, 1, 2, 3]:
    for mm in ['train', 'val']:
        args.mode = mm       # train val
        args.split = sp            # 0 1 2 3
        if args.data_set == 'pascal':
            num_classes = 20
        elif args.data_set == 'coco':
            num_classes = 80

        root_path = args.data_root
        data_path = osp.join(root_path, 'base_annotation/')
        save_path = osp.join(data_path, args.data_set, args.mode, str(args.split))
        check_makedirs(save_path)

        # get class list
        sub_list, sub_val_list = get_train_val_set(args)

        # Locate the standard data list. An explicitly provided --list-root
        # takes precedence over the project-local default.
        if args.list_root is not None:
            list_roots = [osp.abspath(args.list_root)]
        else:
            list_roots = [
                osp.join(project_root, 'lists', args.data_set, 'fss_list'),
            ]

        fss_data_list_path = None
        checked_paths = []
        for list_root in list_roots:
            candidate = osp.join(
                list_root, args.mode, 'data_list_{}.txt'.format(args.split)
            )
            checked_paths.append(candidate)
            if osp.isfile(candidate) and osp.getsize(candidate) > 0:
                fss_data_list_path = candidate
                break

        if fss_data_list_path is None:
            raise FileNotFoundError(
                'Data list not found or empty. Checked:\n  {}'.format(
                    '\n  '.join(checked_paths)
                )
            )

        print('Using data list: {}'.format(fss_data_list_path))
        with open(fss_data_list_path, 'r') as f:
            f_str = f.readlines()
        data_list = []
        for line in f_str:
            line = line.strip()
            if not line:
                continue
            img, mask = line.split(' ', 1)
            data_list.append((img, mask.strip()))

        # Start Processing
        for index in tqdm(range(len(data_list))):
            image_path, label_path = data_list[index]
            # image_path, label_path = root_path + image_path[3:], root_path+ label_path[3:] 
            # print(">>>>>>>>>>>>>>>>>>>>>>>")
            # print(image_path)
            # print(label_path)
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            label_tmp = label.copy()

            for cls in range(1,num_classes+1):
                select_pix = np.where(label_tmp == cls)
                if cls in sub_list:
                    label[select_pix[0],select_pix[1]] = sub_list.index(cls) + 1
                else:
                    label[select_pix[0],select_pix[1]] = 0

            # for pix in np.nditer(label, op_flags=['readwrite']):
            #     if pix == 255:
            #         pass
            #     elif pix not in sub_list: 
            #         pix[...] = 0
            #     else:
            #         pix[...] = sub_list.index(pix) + 1
            
            save_item_path = osp.join(save_path, osp.basename(label_path))
            cv2.imwrite(save_item_path, label)


        print('end')
