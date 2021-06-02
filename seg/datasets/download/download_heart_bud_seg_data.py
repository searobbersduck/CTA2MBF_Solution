import os
import sys
import pandas as pd
import numpy as np

from glob import glob

import fire

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir))
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir, 'external_lib'))

from MedCommon.utils.download_utils import download_dcms_with_website, download_mha_with_csv, get_series_uids, rename_mask_files

def get_heart_hub_seg_series_uids(anno_dir, out_file):
    '''
    anno_dir: '../../../data/cardiac/seg/heart_hub/annotation'
    or anno_dir: '/data/medical/cardiac/seg/heart_hub/annotation'

    tree -L 2
    .
    └── TASK_3836_20201123100128
        ├── image_anno_TASK_3836.csv
        ├── image_component.csv
        ├── series_anno_TASK_3836.csv
        └── series_compoenet.csv

    debug cmd: get_heart_hub_seg_series_uids('../../../data/cardiac/seg/heart_hub/annotation', '../../../data/cardiac/seg/heart_hub/annotation/table/series_uids.txt')
    '''
    anno_files = glob(os.path.join(anno_dir, '*/image_anno_TASK_*.csv'))
    print('====> files processed:\t', anno_files)

    series_uids = []
    for anno_file in anno_files:
        series_uids += get_series_uids(anno_file)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w') as f:
        f.write('\n'.join(series_uids))


def download_images(out_path, config_file):
    '''
    invoke cmd: python download_heart_bud_seg_data.py download_images '../../../data/cardiac/seg/heart_hub/images' '../../../data/cardiac/seg/heart_hub/annotation/table/文件内网地址信息-导出结果_心苞分割.csv'
    '''
    download_dcms_with_website(out_path, config_file)


def download_masks(out_path, anno_dir):
    '''
    anno_dir: '../../../data/cardiac/seg/heart_hub/annotation'
    or anno_dir: '/data/medical/cardiac/seg/heart_hub/annotation'

    tree -L 2
    .
    └── TASK_3836_20201123100128
        ├── image_anno_TASK_3836.csv
        ├── image_component.csv
        ├── series_anno_TASK_3836.csv
        └── series_compoenet.csv

    debug cmd: download_masks('../../../data/cardiac/seg/heart_hub/masks', '../../../data/cardiac/seg/heart_hub/annotation')
    invoke cmd: python download_heart_bud_seg_data.py download_masks '../../../data/cardiac/seg/heart_hub/masks' '../../../data/cardiac/seg/heart_hub/annotation'
    '''
    anno_files = glob(os.path.join(anno_dir, '*/image_anno_TASK_*.csv'))
    print('====> files processed:\t', anno_files)
    for anno_file in anno_files:
        download_mha_with_csv(out_path, anno_file)

def rename_mask_files_local(indir, outdir, anno_dir):
    '''
    debug cmd: rename_mask_files_local('../../../data/cardiac/seg/heart_hub/masks', '../../../data/cardiac/seg/heart_hub/renamed_masks', '../../../data/cardiac/seg/heart_hub/annotation')
    invoke cmd: python download_heart_bud_seg_data.py rename_mask_files_local '../../../data/cardiac/seg/heart_hub/masks' '../../../data/cardiac/seg/heart_hub/renamed_masks' '../../../data/cardiac/seg/heart_hub/annotation'
    '''
    anno_files = glob(os.path.join(anno_dir, '*/image_anno_TASK_*.csv'))
    print('====> files processed:\t', anno_files)
    for anno_file in anno_files:
        rename_mask_files(indir, outdir, anno_file)


if __name__ == '__main__':
    fire.Fire()
    # get_heart_hub_seg_series_uids('../../../data/cardiac/seg/heart_hub/annotation', '../../../data/cardiac/seg/heart_hub/annotation/table/series_uids.txt')



