import os
import sys

from glob import glob
from tqdm import tqdm
import pandas as pd

import pydicom

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(ROOT)

from external_lib.MedCommon.utils.dicom_tag_utils import DicomTagUtils
from external_lib.MedCommon.utils.data_io_utils import DataIO
from external_lib.MedCommon.utils.image_postprocessing_utils import ImagePostProcessingUtils
from external_lib.MedCommon.utils.mask_bounding_utils import MaskBoundingUtils
from external_lib.MedCommon.utils.dicom_tag_utils import DicomTagUtils

from external_lib.MedCommon.experiments.seg.cardiac.chamber_seg.train.train import load_cardic_inference_model
from external_lib.MedCommon.segmentation.runner.train_seg import SegmentationTrainer

# from history.gan.datasets.registration.registration import initial_registration, perform_transform, bspline_registration, bspline_registration_morepoint, registration_three_phase_impl

import shutil
import time
import SimpleITK as sitk
import numpy as np


data_root = '/data/medical/cardiac/cta2mbf/20201216/0.ori'

# 1. 将数据统一调整为如下格式，每个pid下，直接存放dicom文件
'''
    tree -L 1
    .
    ├── 1072995
    ├── 1113550
    ├── 1246311
    ├── 1495136
    ├── 2238598
    ├── 2349198
    ├── 3286359
    ├── 4403780
    ├── 5142256
    ├── 5154253
    └── 5172816

'''

'''
    tree -L 1
        .
        ├── CTAMBF-11例
        ├── CTAMBF-14例
        ├── CTAMBF-26例
        └── 心肌灌注

    该文件夹为最原始的存放方式，其中“CTAMBF-11例”和“CTAMBF-26例”的存放基本符合要求， “CTAMBF-14例”和“心肌灌注”要进行调整


    CTAMBF-11例$ tree -L 1
        .
        ├── 1072995
        ├── 1113550
        ├── 1246311
        ├── 1495136
        ├── 2238598
        ├── 2349198
        ├── 3286359
        ├── 4403780
        ├── 5142256
        ├── 5154253
        └── 5172816



    CTAMBF-26例$ tree -L 1
        .
        ├── 1176296
        ├── 1180514
        ├── 1282840
        ├── 1406241
        ├── 1591036
        ├── 2320047
        ├── 2353593
        ├── 2580944
        ├── 2676146
        ├── 2687597
        ├── 3334300
        ├── 3797295
        ├── 3797878
        ├── 4228228
        ├── 4254469
        ├── 4279440
        ├── 4383280
        ├── 4520118
        ├── 4611272
        ├── 4620137
        ├── 4646679
        ├── 4916091
        ├── 4947026
        ├── 5154651
        ├── 5230924
        └── 5234999



    心肌灌注$ tree -L 4
        .
        ├── 1200335
        │   ├── DICOM
        │   │   └── 20102621
        │   │       ├── 18170000
        │   │       ├── 18170001
        │   │       ├── 18170002
        │   │       ├── 18170003
        │   │       ├── 18170004
        │   │       ├── 18170005
        │   │       ├── 18170006
        │   │       └── 18170007
        │   └── DICOMDIR
        ├── 1979010
        │   ├── DICOM
        │   │   └── 20102620
        │   │       ├── 59400000
        │   │       ├── 59400001
        │   │       ├── 59400002
        │   │       ├── 59400003
        │   │       ├── 59400004
        │   │       └── 59400005
        │   └── DICOMDIR
        ├── 2360212
        │   ├── DICOM
        │   │   └── 20102621
        │   │       ├── 30100000
        │   │       ├── 30100001
        │   │       ├── 30100002
        │   │       ├── 30100003
        │   │       ├── 30100004
        │   │       ├── 30100005
        │   │       ├── 30100006
        │   │       ├── 42320000
        │   │       ├── 42320001
        │   │       ├── 42320002
        │   │       ├── 42320003
        │   │       ├── 42320004
        │   │       ├── 42320005
        │   │       └── 42320006
        │   └── DICOMDIR
        ├── 2575877
        │   ├── DICOM
        │   │   └── 20102620
        │   │       ├── 55560000
        │   │       ├── 55560001
        │   │       ├── 55560002
        │   │       ├── 55560003
        │   │       ├── 55560004
        │   │       └── 55560005
        │   └── DICOMDIR
        ├── 4804596
        │   ├── DICOM
        │   │   └── 20102621
        │   │       ├── 12210000
        │   │       ├── 12210001
        │   │       ├── 12210002
        │   │       ├── 12210003
        │   │       ├── 12210004
        │   │       ├── 12210005
        │   │       ├── 12210006
        │   │       └── 12210007
        │   └── DICOMDIR
        ├── 5179724
        │   ├── DICOM
        │   │   └── 20102621
        │   │       ├── 03330000
        │   │       ├── 03330001
        │   │       ├── 03330002
        │   │       ├── 03330003
        │   │       ├── 03330004
        │   │       └── 03330005
        │   └── DICOMDIR
        ├── 5188097
        │   ├── DICOM
        │   │   └── 20102621
        │   │       ├── 23540000
        │   │       ├── 23540001
        │   │       ├── 23540002
        │   │       ├── 23540003
        │   │       ├── 23540004
        │   │       ├── 23540005
        │   │       ├── 23540006
        │   │       ├── 23540007
        │   │       ├── 23540008
        │   │       ├── 36020000
        │   │       ├── 36020001
        │   │       ├── 36020002
        │   │       ├── 36020003
        │   │       ├── 36020004
        │   │       ├── 36020005
        │   │       ├── 36020006
        │   │       ├── 36020007
        │   │       └── 36020008
        │   └── DICOMDIR
        ├── 5188555
        │   ├── DICOM
        │   │   └── 20102621
        │   │       ├── 35020000
        │   │       ├── 35020001
        │   │       ├── 47380000
        │   │       ├── 47380001
        │   │       ├── 47380002
        │   │       ├── 47380003
        │   │       ├── 47380004
        │   │       ├── 47380005
        │   │       ├── 47380006
        │   │       └── 47380007
        │   └── DICOMDIR
        └── 5195931
            ├── DICOM
            │   └── 20102621
            │       ├── 07160000
            │       ├── 07160001
            │       ├── 07160002
            │       ├── 07160003
            │       ├── 07160004
            │       ├── 07160005
            │       ├── 07160006
            │       └── 07160007
            └── DICOMDIR


    CTAMBF-14例$ tree -L 4
        .
        ├── 1069558
        │   ├── DICOM
        │   │   └── 20112001
        │   │       ├── 21140000
        │   │       ├── 21140001
        │   │       ├── 21140002
        │   │       ├── 21140003
        │   │       ├── 21140004
        │   │       ├── 21140005
        │   │       ├── 21140006
        │   │       └── 21140007
        │   └── DICOMDIR
        ├── 1315171
        │   ├── DICOM
        │   │   └── 20111924
        │   │       ├── 17210000
        │   │       ├── 17210001
        │   │       ├── 17210002
        │   │       ├── 17210003
        │   │       └── 17210004
        │   ├── DICOMDIR
        │   └── New Folder
        │       └── 1989267
        ├── 1668953
        │   ├── DICOM
        │   │   └── 20112001
        │   │       ├── 14050000
        │   │       ├── 14050001
        │   │       ├── 14050002
        │   │       ├── 14050003
        │   │       ├── 14050004
        │   │       └── 14050005
        │   └── DICOMDIR
        ├── 1723746
        │   ├── DICOM
        │   │   └── 20111924
        │   │       ├── 22350000
        │   │       ├── 22350001
        │   │       ├── 22350002
        │   │       ├── 22350003
        │   │       ├── 22350004
        │   │       ├── 22350005
        │   │       ├── 22350006
        │   │       ├── 22350007
        │   │       ├── 22350008
        │   │       └── 22350009
        │   └── DICOMDIR
        ├── 1989267
        │   ├── DICOM
        │   │   └── 20112001
        │   │       ├── 03190000
        │   │       ├── 03190001
        │   │       ├── 03190002
        │   │       ├── 03190003
        │   │       ├── 03190004
        │   │       ├── 03190005
        │   │       ├── 03190006
        │   │       └── 03190007
        │   └── DICOMDIR
        ├── 2068282
        │   ├── DICOM
        │   │   └── 20112001
        │   │       ├── 25440000
        │   │       ├── 25440001
        │   │       ├── 25440002
        │   │       ├── 25440003
        │   │       ├── 25440004
        │   │       └── 25440005
        │   └── DICOMDIR
        ├── 2146500
        │   ├── DICOM
        │   │   └── 20112001
        │   │       ├── 07440000
        │   │       ├── 07440001
        │   │       ├── 07440002
        │   │       ├── 07440003
        │   │       ├── 07440004
        │   │       ├── 07440005
        │   │       ├── 07440006
        │   │       ├── 07440007
        │   │       ├── 07440008
        │   │       ├── 07440009
        │   │       └── 07440010
        │   └── DICOMDIR
        ├── 2418721
        │   ├── DICOM
        │   │   └── 20111924
        │   │       ├── 39270000
        │   │       ├── 39270001
        │   │       ├── 39270002
        │   │       ├── 39270003
        │   │       ├── 39270004
        │   │       ├── 39270005
        │   │       └── 39270006
        │   └── DICOMDIR
        ├── 2978559
        │   ├── DICOM
        │   │   └── 20111924
        │   │       ├── 32590000
        │   │       ├── 32590001
        │   │       ├── 32590002
        │   │       ├── 32590003
        │   │       ├── 32590004
        │   │       └── 32590005
        │   └── DICOMDIR
        ├── 4000653
        │   ├── DICOM
        │   │   └── 20111924
        │   │       ├── 58200000
        │   │       ├── 58200001
        │   │       ├── 58200002
        │   │       ├── 58200003
        │   │       ├── 58200004
        │   │       ├── 58200005
        │   │       ├── 58200006
        │   │       └── 58200007
        │   └── DICOMDIR
        ├── 4081140
        │   ├── DICOM
        │   │   └── 20111924
        │   │       ├── 47390000
        │   │       ├── 47390001
        │   │       ├── 47390002
        │   │       ├── 47390003
        │   │       ├── 47390004
        │   │       ├── 47390005
        │   │       ├── 47390006
        │   │       └── 47390007
        │   └── DICOMDIR
        ├── 4977919
        │   ├── DICOM
        │   │   └── 20112001
        │   │       ├── 33070000
        │   │       ├── 33070001
        │   │       ├── 33070002
        │   │       ├── 33070003
        │   │       ├── 33070004
        │   │       └── 33070005
        │   └── DICOMDIR
        ├── 5073243
        │   ├── DICOM
        │   │   └── 20112001
        │   │       ├── 29160000
        │   │       ├── 29160001
        │   │       ├── 29160002
        │   │       ├── 29160003
        │   │       ├── 29160004
        │   │       └── 29160005
        │   └── DICOMDIR
        └── 5211311
            ├── DICOM
            │   └── 20112001
            │       ├── 17240000
            │       ├── 17240001
            │       ├── 17240002
            │       ├── 17240003
            │       ├── 17240004
            │       ├── 17240005
            │       └── 17240006
            └── DICOMDIR

'''
def copy_from_folder_format1(src_root, dst_root):
    '''
        CTAMBF-11例$ tree -L 1
        .
        ├── 1072995
        ├── 1113550
        ├── 1246311
        ├── 1495136
        ├── 2238598
        ├── 2349198
        ├── 3286359
        ├── 4403780
        ├── 5142256
        ├── 5154253
        └── 5172816
    '''
    os.makedirs(dst_root, exist_ok=True)
    pids = os.listdir(src_root)
    for pid in pids:
        if len(pid) != 7:
            continue
        src_file = os.path.join(src_root, pid)
        dst_file = os.path.join(dst_root, pid)
        shutil.copytree(src_file, dst_file)
        print('copy from {} to {}'.format(src_file, dst_file))


def copy_from_folder_format2(src_root, dst_root):
    '''
        心肌灌注$ tree -L 4
        .
        ├── 1200335
        │   ├── DICOM
        │   │   └── 20102621
        │   │       ├── 18170000
        │   │       ├── 18170001
        │   │       ├── 18170002
        │   │       ├── 18170003
        │   │       ├── 18170004
        │   │       ├── 18170005
        │   │       ├── 18170006
        │   │       └── 18170007
        │   └── DICOMDIR
        ├── 1979010
        ...
    '''
    os.makedirs(dst_root, exist_ok=True)
    pids = os.listdir(src_root)
    for pid in pids:
        if len(pid) != 7:
            continue
        dst_pid_root = os.path.join(dst_root, pid)
        os.makedirs(dst_pid_root, exist_ok=True)
        dicom_folder = os.path.join(src_root, pid, 'DICOM')
        what_ids = os.listdir(dicom_folder)
        what_id = what_ids[0]
        dicom_inner_folder = os.path.join(dicom_folder, what_id)
        fuck_ids = os.listdir(dicom_inner_folder)
        for fuck_id in fuck_ids:
            fuck_path = os.path.join(dicom_inner_folder, fuck_id)
            for f in os.listdir(fuck_path):
                src_file = os.path.join(fuck_path, f)
                dst_file = os.path.join(dst_pid_root, f)
                shutil.copyfile(src_file, dst_file)
                print('====> cropy from {} to {}'.format(src_file, dst_file))


def step_1_crop_from_ori1_to_ori2():
    '''
        tree -L 1
        .
        ├── CTAMBF-11例
        ├── CTAMBF-14例
        ├── CTAMBF-26例
        └── 心肌灌注
    '''
    data_root = '/data/medical/cardiac/cta2mbf/20201216/0.ori'
    dst_root = '/data/medical/cardiac/cta2mbf/20201216/0.ori_2'
    copy_from_folder_format1(os.path.join(data_root, 'CTAMBF-11例'), dst_root)
    copy_from_folder_format1(os.path.join(data_root, 'CTAMBF-26例'), dst_root)
    copy_from_folder_format2(os.path.join(data_root, 'CTAMBF-14例'), dst_root)
    copy_from_folder_format2(os.path.join(data_root, '心肌灌注'), dst_root)


# 2. 将每个文件夹下直接存放dicom文件的格式，转变为按series uid进行分子文件夹存放
def sort_by_series_uid_onecase(in_pid_root, out_pid_root):
    os.makedirs(out_pid_root, exist_ok=True)
    dicom_files = glob(os.path.join(in_pid_root, '*'))
    cnt_i = 0
    for dicom_file in tqdm(dicom_files):
        try:
            meta_data = pydicom.read_file(dicom_file)
            series_uid = meta_data.SeriesInstanceUID
            dst_suid_root = os.path.join(out_pid_root, series_uid)
            os.makedirs(dst_suid_root, exist_ok=True)
            src_file = dicom_file
            dst_file = os.path.join(dst_suid_root, '{}.dcm'.format(cnt_i))
            shutil.copyfile(src_file, dst_file)
            cnt_i += 1
        except:
            pass

def sort_by_series_uid_singletask(pids, in_root, out_root):
    os.makedirs(out_root, exist_ok=True)
    for pid in tqdm(pids):
        in_pid_root = os.path.join(in_root, pid)
        out_pid_root = os.path.join(out_root, os.path.basename(pid))
        sort_by_series_uid_onecase(in_pid_root, out_pid_root)


def sort_by_series_uid_multiprocessing(in_root, out_root, process_num=12):
    pids = os.listdir(in_root)

    num_per_process = (len(pids) + process_num - 1)//process_num

    # this for single thread to debug
    # sort_by_series_uid_singletask(pids, in_root, out_root)

    # this for run 
    import multiprocessing
    from multiprocessing import Process
    multiprocessing.freeze_support()

    pool = multiprocessing.Pool()

    results = []

    # print(len(pids))
    for i in range(process_num):
        sub_pids = pids[num_per_process*i:min(num_per_process*(i+1), len(pids))]
        print(len(sub_pids))
        result = pool.apply_async(sort_by_series_uid_singletask, 
            args=(sub_pids, in_root, out_root))
        results.append(result)

    pool.close()
    pool.join()

def step_2_sort_by_series_uid():
    in_root = '/data/medical/cardiac/cta2mbf/20201216/0.ori_2'
    out_root = '/data/medical/cardiac/cta2mbf/20201216/0.ori_3'
    sort_by_series_uid_multiprocessing(in_root, out_root)


# 3. 提取出CTA/CTP/MIP/AVG序列
'''
对于CTAMBF-11例、CTAMBF-26例两个文件夹中的数据
1072995
    ImageType: DERIVED\SECONDARY\AXIAL\CT_SOM7 VPCT\AVG     1.3.12.2.1107.5.8.15.130931.30000020100515452825700018288
    ImageType: DERIVED\SECONDARY\AXIAL\CT_SOM7 VPCT\MIP     1.3.12.2.1107.5.8.15.130931.30000020100515452825700018235
    ImageType: DERIVED\SECONDARY\AXIAL\CT_SOM7 VPCT\BF      1.3.12.2.1107.5.8.15.130931.30000020100515452825700018341


    ImageType: DERIVED\SECONDARY\AXIAL\CT_SOM8 PERF\MIP     1.3.12.2.1107.5.8.15.130931.30000020110414325175800001722   1.2.194.0.108707908.20201104000102.1817.12100.21624123 
    ImageType: DERIVED\SECONDARY\AXIAL\CT_SOM8 PERF\MIP     1.3.12.2.1107.5.8.15.130931.30000020110414401532500002053   1.2.194.0.108707908.20201104000102.1817.12100.21624123
    ImageType: DERIVED\SECONDARY\AXIAL\CT_SOM8 PERF\AVG     1.3.12.2.1107.5.8.15.130931.30000020110414325175800001776   1.2.194.0.108707908.20201104000102.1817.12100.21624123
    ImageType: DERIVED\SECONDARY\AXIAL\CT_SOM8 PERF\AVG     1.3.12.2.1107.5.8.15.130931.30000020110414401532500002159   1.2.194.0.108707908.20201104000102.1817.12100.21624123 
'''

def extract_ctp_related(data_root, out_root):
    os.makedirs(out_root, exist_ok=True)
    pids = os.listdir(data_root)
    for pid in tqdm(pids):
        if len(pid) != 7:
            continue
        print('\n', pid)
        pid_root = os.path.join(data_root, pid)
        mip_cnt = 0
        avg_cnt = 0
        bf_cnt = 0
        mip_src_files = []
        avg_src_files = []
        bf_src_files = []
        for suid in os.listdir(pid_root):
            if len(suid) < 10:
                continue
            series_path = os.path.join(pid_root, suid)
            try:
                dicom_files = glob(os.path.join(series_path, '*'))
                if len(dicom_files) < 20:
                    continue
                dicom_file = dicom_files[0]
                meta_data = pydicom.read_file(dicom_file)
                if 'MIP' in meta_data.ImageType:
                    mip_cnt += 1
                    print(suid)
                    print(meta_data.ImageType, '\t', meta_data.StudyInstanceUID, '\t', len(dicom_files))
                    mip_src_files.append(series_path)
                if 'AVG' in meta_data.ImageType:
                    avg_cnt += 1
                    print(suid)
                    print(meta_data.ImageType, '\t', meta_data.StudyInstanceUID, '\t', len(dicom_files))
                    avg_src_files.append(series_path)
                if 'BF' in meta_data.ImageType:
                    bf_cnt += 1
                    print(suid)
                    print(meta_data.ImageType, '\t', meta_data.StudyInstanceUID, '\t', len(dicom_files))
                    bf_src_files.append(series_path)
            except:
                pass
        if len(mip_src_files) == 1 and len(avg_src_files) == 1 and len(bf_src_files):
            dst_pid_root = os.path.join(out_root, pid)
            os.makedirs(dst_pid_root, exist_ok=True)
            src_mip_file = mip_src_files[0]
            dst_mip_file = os.path.join(dst_pid_root, 'MIP', os.path.basename(src_mip_file))
            src_avg_file = avg_src_files[0]
            dst_avg_file = os.path.join(dst_pid_root, 'AVG', os.path.basename(src_avg_file))
            src_bf_file = bf_src_files[0]
            dst_bf_file = os.path.join(dst_pid_root, 'BF', os.path.basename(src_bf_file))
            shutil.copytree(src_mip_file, dst_mip_file)
            shutil.copytree(src_avg_file, dst_avg_file)
            shutil.copytree(src_bf_file, dst_bf_file)

def step_3_1_extract_ctp_related():
    data_root = '/data/medical/cardiac/cta2mbf/20201216/0.ori_3'
    out_root = '/data/medical/cardiac/cta2mbf/20201216/3.sorted'
    extract_ctp_related(data_root, out_root)

def extract_cta_file(data_root, cta_config_file, out_root):
    df = pd.read_csv(cta_config_file)
    for index, row in tqdm(df.iterrows()):
        pid, cta_series = row
        pid = str(pid)
        dst_pid_root = os.path.join(out_root, pid)
        # todo: 当前CTP数据不全，补全后，应该忽略下边的排除条件
        if not os.path.isdir(dst_pid_root):
            continue
        
        src_cta_file = os.path.join(data_root, pid, cta_series)
        if not os.path.isdir(src_cta_file):
            print(src_cta_file)
            continue
        dst_cta_file = os.path.join(dst_pid_root, 'CTA', cta_series)
        shutil.copytree(src_cta_file, dst_cta_file)

def step_3_2_extract_cta_file():
    data_root = '/data/medical/cardiac/cta2mbf/20201216/0.ori_3'
    cta_config_file = '/data/medical/cardiac/cta2mbf/20201216/0.config_info/cta_series.csv'
    out_root = '/data/medical/cardiac/cta2mbf/20201216/3.sorted'
    extract_cta_file(data_root, cta_config_file, out_root)

def test_extract_ctp_related():
    data_root = '/data/medical/cardiac/cta2mbf/20201216/0.ori_3'
    out_root = '/data/medical/cardiac/cta2mbf/20201216/0.ori_4'
    extract_ctp_related(data_root, out_root)

# 老算法，目前不可用
def cardiac_segmentation(data_root = '/data/medical/cardiac/cta2mbf/20201216/3.sorted', 
    out_dir = '/data/medical/cardiac/cta2mbf/20201216/3.sorted_mask'):

    import torch
    import torch.nn

    # data_root = '/data/medical/cardiac/cta2mbf/20201216/3.sorted'
    # out_dir = '/data/medical/cardiac/cta2mbf/20201216/3.sorted_mask'
    # out_cta_dir = '/data/medical/cardiac/cta2mbf/20201216/3.sorted_mask/CTA'
    
    # os.makedirs(out_cta_dir, exist_ok=True)

    # load seg model
    model_pth = '../external_lib/MedCommon/experiments/seg/cardiac/chamber_seg/train/common_seg_train_0.030_val_0.055'
    model = load_cardic_inference_model(model_pth)
    model = torch.nn.DataParallel(model).cuda()
    model.eval()
    
    for pid in tqdm(os.listdir(data_root)):
        pid_path = os.path.join(data_root, pid)
        if not os.path.isdir(pid_path):
            print('patient path not exist!\t{}'.format(pid_path))
            continue
        cta_root = os.path.join(pid_path, 'CTA')
        cta_files = os.listdir(cta_root)
        cta_file = cta_files[0]
        cta_file = os.path.join(cta_root, cta_file)
        if not os.path.isdir(cta_file):
            print('cta file not exist!\t{}'.format(cta_file))
            continue
        image, pred_mask = SegmentationTrainer.inference_one_case(model, cta_file, is_dcm=True)

        out_cta_dir = os.path.join(out_dir, pid, 'CTA')
        os.makedirs(out_cta_dir, exist_ok=True)
        out_cta_file = os.path.join(out_cta_dir, 'CTA.nii.gz')
        out_cta_mask_file = os.path.join(out_cta_dir, 'CTA_MASK.nii.gz')

        sitk.WriteImage(image, out_cta_file)
        sitk.WriteImage(pred_mask, out_cta_mask_file)

    
    # for pid in tqdm(os.listdir(data_root)):
    #     pid_path = os.path.join(data_root, pid)
    #     if not os.path.isdir(pid_path):
    #         print('patient path not exist!\t{}'.format(pid_path))
    #         continue
    #     cta_root = os.path.join(pid_path, 'MIP')
    #     cta_files = os.listdir(cta_root)
    #     cta_file = cta_files[0]
    #     cta_file = os.path.join(cta_root, cta_file)
    #     if not os.path.isdir(cta_file):
    #         print('cta file not exist!\t{}'.format(cta_file))
    #         continue
    #     image, pred_mask = SegmentationTrainer.inference_one_case(model, cta_file, is_dcm=True)

    #     out_cta_dir = os.path.join(out_dir, pid, 'MIP')
    #     os.makedirs(out_cta_dir, exist_ok=True)
    #     out_cta_file = os.path.join(out_cta_dir, 'MIP.nii.gz')
    #     out_cta_mask_file = os.path.join(out_cta_dir, 'MIP_MASK.nii.gz')

    #     sitk.WriteImage(image, out_cta_file)
    #     sitk.WriteImage(pred_mask, out_cta_mask_file)


    for pid in tqdm(os.listdir(data_root)):
        pid_path = os.path.join(data_root, pid)
        if not os.path.isdir(pid_path):
            print('patient path not exist!\t{}'.format(pid_path))
            continue
        cta_root = os.path.join(pid_path, 'AVG')
        cta_files = os.listdir(cta_root)
        cta_file = cta_files[0]
        cta_file = os.path.join(cta_root, cta_file)
        if not os.path.isdir(cta_file):
            print('cta file not exist!\t{}'.format(cta_file))
            continue
        image, pred_mask = SegmentationTrainer.inference_one_case(model, cta_file, is_dcm=True)

        out_cta_dir = os.path.join(out_dir, pid, 'AVG')
        os.makedirs(out_cta_dir, exist_ok=True)
        out_cta_file = os.path.join(out_cta_dir, 'AVG.nii.gz')
        out_cta_mask_file = os.path.join(out_cta_dir, 'AVG_MASK.nii.gz')

        sitk.WriteImage(image, out_cta_file)
        sitk.WriteImage(pred_mask, out_cta_mask_file)

def cardiac_segmentation_new_algo(
        data_root=None, 
        out_dir = None
    ):
    import torch
    from external_lib.MedCommon.experiments.seg.cardiac.chamber.inference.inference import load_inference_opts
    from external_lib.MedCommon.segmentation.runner.train_seg import SegmentationTrainer
    opts = load_inference_opts()
    model = SegmentationTrainer.load_model(opts)
    model = torch.nn.DataParallel(model).cuda()
    model.eval()

    for pid in tqdm(os.listdir(data_root)):
        pid_path = os.path.join(data_root, pid)
        if not os.path.isdir(pid_path):
            print('patient path not exist!\t{}'.format(pid_path))
            continue
        cta_root = os.path.join(pid_path, 'CTA')
        cta_files = os.listdir(cta_root)
        cta_file = cta_files[0]
        cta_file = os.path.join(cta_root, cta_file)
        if not os.path.isdir(cta_file):
            print('cta file not exist!\t{}'.format(cta_file))
            continue
        image, pred_mask = SegmentationTrainer.inference_one_case(model, cta_file, is_dcm=True)

        out_cta_dir = os.path.join(out_dir, pid, 'CTA')
        os.makedirs(out_cta_dir, exist_ok=True)
        out_cta_file = os.path.join(out_cta_dir, 'CTA.nii.gz')
        out_cta_mask_file = os.path.join(out_cta_dir, 'CTA_MASK.nii.gz')

        sitk.WriteImage(image, out_cta_file)
        sitk.WriteImage(pred_mask, out_cta_mask_file)    
    


def step_3_3_segment_cardiac():
    cardiac_segmentation()

def step_3_3_segment_cardiac_connected_region(root_dir = '/data/medical/cardiac/cta2mbf/20201216/3.sorted_mask'):
    # root_dir = '/data/medical/cardiac/cta2mbf/20201216/3.sorted_mask'
    for pid in tqdm(os.listdir(root_dir)):
        pid_path = os.path.join(root_dir, pid)
        if not os.path.isdir(pid_path):
            continue
        cta_root = os.path.join(pid_path, 'CTA')
        mip_root = os.path.join(pid_path, 'MIP')
        avg_root = os.path.join(pid_path, 'AVG')
        
        in_cta_file = os.path.join(cta_root, 'CTA_MASK.nii.gz')
        out_cta_file = os.path.join(cta_root, 'CTA_MASK_connected.nii.gz')
        in_mip_file = os.path.join(mip_root, 'MIP_MASK.nii.gz')
        out_mip_file = os.path.join(mip_root, 'MIP_MASK_connected.nii.gz')
        in_avg_file = os.path.join(avg_root, 'AVG_MASK.nii.gz')
        out_avg_file = os.path.join(avg_root, 'AVG_MASK_connected.nii.gz')

        try:
            if os.path.isfile(in_cta_file):
                in_mask = sitk.ReadImage(in_cta_file)
                out_mask_sitk = ImagePostProcessingUtils.get_maximal_connected_region_multilabel(in_mask, mask_labels=[1, 2, 3, 4, 6])
                sitk.WriteImage(out_mask_sitk, out_cta_file)

            if os.path.isfile(in_mip_file):
                in_mask = sitk.ReadImage(in_mip_file)
                out_mask_sitk = ImagePostProcessingUtils.get_maximal_connected_region_multilabel(in_mask, mask_labels=[1, 2, 3, 4, 6])
                sitk.WriteImage(out_mask_sitk, out_mip_file)

            if os.path.isfile(in_avg_file):
                in_mask = sitk.ReadImage(in_avg_file)
                out_mask_sitk = ImagePostProcessingUtils.get_maximal_connected_region_multilabel(in_mask, mask_labels=[1, 2, 3, 4, 6])
                sitk.WriteImage(out_mask_sitk, out_avg_file)
        except Exception as e:
            print(e)
            print('====> Error case:\t{}'.format(pid))

def step_3_4_convert_dicom_series_to_nii(data_root = '/data/medical/cardiac/cta2mbf/20201216/3.sorted', out_dir = '/data/medical/cardiac/cta2mbf/20201216/3.sorted_nii'):
    # data_root = '/data/medical/cardiac/cta2mbf/20201216/3.sorted'
    # out_dir = '/data/medical/cardiac/cta2mbf/20201216/3.sorted_nii'
    
    os.makedirs(out_dir, exist_ok=True)

    for pid in tqdm(os.listdir(data_root)):
        try:
            in_pid_path = os.path.join(data_root, pid)
            out_pid_path = os.path.join(out_dir, pid)
            os.makedirs(out_pid_path, exist_ok=True)
            
            in_cta_path = os.path.join(in_pid_path, 'CTA')
            in_cta_path = glob(os.path.join(in_cta_path,'*'))[0]
            in_mip_path = os.path.join(in_pid_path, 'MIP')
            in_mip_path = glob(os.path.join(in_mip_path, '*'))[0]
            in_avg_path = os.path.join(in_pid_path, 'AVG')
            in_avg_path = glob(os.path.join(in_avg_path, '*'))[0]
            in_bf_path = os.path.join(in_pid_path, 'BF')
            in_bf_path = glob(os.path.join(in_bf_path, '*'))[0]

            out_cta_path = os.path.join(out_pid_path, 'CTA.nii.gz')
            out_mip_path = os.path.join(out_pid_path, 'MIP.nii.gz')
            out_avg_path = os.path.join(out_pid_path, 'AVG.nii.gz')
            out_bf_path = os.path.join(out_pid_path, 'BF.nii.gz')

            cta_img = DataIO.load_dicom_series(in_cta_path)['sitk_image']
            mip_img = DataIO.load_dicom_series(in_mip_path)['sitk_image']
            avg_img = DataIO.load_dicom_series(in_avg_path)['sitk_image']
            bf_img = DataIO.load_dicom_series(in_bf_path)['sitk_image']

            sitk.WriteImage(cta_img, out_cta_path)
            sitk.WriteImage(mip_img, out_mip_path)
            sitk.WriteImage(avg_img, out_avg_path)
            sitk.WriteImage(bf_img, out_bf_path)

        except:
            pass

def register_images(cta_file, mip_file, bf_file, is_dcm=False):
    if is_dcm:
        cta_data = DataIO.load_dicom_series(cta_file)
        mip_data = DataIO.load_dicom_series(mip_file)
        bf_data = DataIO.load_dicom_series(bf_file)
    else:
        cta_data = DataIO.load_nii_image(cta_file)
        mip_data = DataIO.load_nii_image(mip_file)
        bf_data = DataIO.load_nii_image(bf_file)
    cta_img = cta_data['sitk_image']
    mip_img = mip_data['sitk_image']
    bf_img = bf_data['sitk_image']

    selx = sitk.ElastixImageFilter()
    selx.SetFixedImage(cta_img)
    selx.SetMovingImage(mip_img)
    selx.SetParameterMap(selx.GetDefaultParameterMap('nonrigid'))
    selx.Execute()

    moved_mip_img = sitk.Transformix(mip_img, selx.GetTransformParameterMap())

    sitk.WriteImage(moved_mip_img, 'selx.nii.gz')

def rigid_register_images(cta_file, mip_file, bf_file, out_dir, is_dcm=False):
    if is_dcm:
        cta_data = DataIO.load_dicom_series(cta_file)
        mip_data = DataIO.load_dicom_series(mip_file)
        bf_data = DataIO.load_dicom_series(bf_file)
    else:
        cta_data = DataIO.load_nii_image(cta_file)
        mip_data = DataIO.load_nii_image(mip_file)
        bf_data = DataIO.load_nii_image(bf_file)
    cta_img = cta_data['sitk_image']
    mip_img = mip_data['sitk_image']
    bf_img = bf_data['sitk_image']

    fixed_image, mip_ffd, mbf_ffd, mip_ffd_morepoint, mbf_ffd_morepoint, mip_ffd_ffd, mbf_ffd_ffd = registration_three_phase_impl(cta_img, mip_img, bf_img)

    os.makedirs(out_dir, exist_ok=True)

    sitk.WriteImage(fixed_image, os.path.join(out_dir, 'cta_image.nii.gz'))
    sitk.WriteImage(mip_ffd, os.path.join(out_dir, 'mip_ffd.nii.gz'))
    sitk.WriteImage(mbf_ffd, os.path.join(out_dir, 'mbf_ffd.nii.gz'))
    sitk.WriteImage(mip_ffd_morepoint, os.path.join(out_dir, 'mip_ffd_morepoint.nii.gz'))
    sitk.WriteImage(mbf_ffd_morepoint, os.path.join(out_dir, 'mbf_ffd_morepoint.nii.gz'))
    sitk.WriteImage(mip_ffd_ffd, os.path.join(out_dir, 'mip_ffd_ffd.nii.gz'))
    sitk.WriteImage(mbf_ffd_ffd, os.path.join(out_dir, 'mbf_ffd_ffd.nii.gz'))

def test_register_images():
    pid = '1315171'
    data_root = '/data/medical/cardiac/cta2mbf/20201216/3.sorted/'
    pid_path = os.path.join(data_root, pid)
    cta_dir = os.path.join(pid_path, 'CTA')
    cta_files = os.listdir(cta_dir)
    cta_file = os.path.join(cta_dir, cta_files[0])
    mip_dir = os.path.join(pid_path, 'AVG')
    mip_files = os.listdir(mip_dir)
    mip_file = os.path.join(mip_dir, mip_files[0])
    bf_dir = os.path.join(pid_path, 'BF')
    bf_files = os.listdir(bf_dir)
    bf_file = os.path.join(bf_dir, bf_files[0])
    
    # cta_file = '/data/medical/cardiac/cta2mbf/20201216/3.sorted/1072995/CTA/1.3.12.2.1107.5.8.15.130931.30000020100515452825700016870'
    # mip_file = '/data/medical/cardiac/cta2mbf/20201216/3.sorted/1072995/MIP/1.3.12.2.1107.5.8.15.130931.30000020100515452825700018235'
    # bf_file = '/data/medical/cardiac/cta2mbf/20201216/3.sorted/1072995/BF/1.3.12.2.1107.5.8.15.130931.30000020100515452825700018341'

    out_dir = '/data/medical/cardiac/cta2mbf/20201216/4.registration_test/{}'.format(pid)

    beg = time.time()
    # register_images(cta_file, mip_file, bf_file, True)
    rigid_register_images(cta_file, mip_file, bf_file, out_dir, True)
    end = time.time()
    print('====> test_register_images time cosume is:\t{:.3f}'.format(end-beg))

def generate_register_cmd(
        data_root = '/data/zhangwd/data/cardiac/3.sorted_nii',
        out_file = '/data/zhangwd/data/cardiac/3.sorted_nii_config/registration_mbf2cta.sh',
        elastic_path = '/data/hjy/elastix-5.0/elastix',
        param1_path = '/data/hjy/elastix-5.0/params_copd/parameters_Affine.txt',
        param2_path = '/data/hjy/elastix-5.0/params_copd/parameters_BSpline.txt',
        param3_path = '/data/hjy/elastix-5.0/params_copd/parameters_Rigid.txt'):
    # data_root = '/data/medical/cardiac/cta2mbf/20201216/3.sorted_nii'
    # data_root = '/data/zhangwd/data/cardiac/3.sorted_nii'
    # out_file = '/data/zhangwd/data/cardiac/3.sorted_nii_config/registration_mbf2cta.sh'
    # elastic_path = '/data/hjy/elastix-5.0/elastix'
    # param1_path = '/data/hjy/elastix-5.0/params_copd/parameters_Affine.txt'
    # param2_path = '/data/hjy/elastix-5.0/params_copd/parameters_BSpline.txt'
    # param3_path = '/data/hjy/elastix-5.0/params_copd/parameters_Rigid.txt'

    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    cmd_pattern = '{} -f {} -m {} -out {} -p {} -p {} -p {}'
    cp_pattern = 'mv {} {} -v'
    rm_pattern = 'rm -rf {}'
    
    cmd_list = []
    cp_list = []
    rm_list = []
    for pid in tqdm(os.listdir(data_root)):
        patient_path = os.path.join(data_root, pid)
        fixed_path = os.path.join(patient_path, 'CTA.nii.gz')
        moving_path = os.path.join(patient_path, 'MIP.nii.gz')
        out_path = os.path.join(patient_path, 'result')
        os.makedirs(out_path, exist_ok=True)

        cmd = cmd_pattern.format(elastic_path, fixed_path, moving_path, out_path, param1_path, param2_path, param3_path)
        cp_cmd = cp_pattern.format(os.path.join(out_path, 'result.2.mha'), os.path.join(patient_path, 'registration_bf_mip.mha'))
        rm_cmd = rm_pattern.format(os.path.join(out_path))

        cmd_list.append(cmd)
        cp_list.append(cp_cmd)
        rm_list.append(rm_cmd)

    with open(out_file, 'w') as f:
        f.write('\n'.join(cmd_list))
        f.write('\n'.join(cp_list))
        f.write('\n'.join(rm_list))
    
def check_last_5():
    '''
    验证通过序列号的后五位字符，能否做到和序列号一一对应，结论是不可以
    '''
    data_root = '/data/medical/cardiac/cta2mbf/20201216/0.ori_3'
    suids_list = []
    for pid in tqdm(os.listdir(data_root)):
        patient_path = os.path.join(data_root, pid)
        # suids = os.listdir(patient_path)
        suids = []
        for uid in os.listdir(patient_path):
            uid_path = os.path.join(patient_path, uid)
            dcm_files = os.listdir(uid_path)
            if len(dcm_files) < 10:
                continue
            suids.append(uid)
        suids_list += suids

    last_5_suids_list = [i[-5:] for i in suids_list]
    last_5_suids_list = list(set(last_5_suids_list))

    print('hello world!')

def step_5_1_extract_mbf_myocardium(
        mbf_root = '/data/medical/cardiac/cta2mbf/20201216/4.registration_batch1',
        mask_root = '/data/medical/cardiac/cta2mbf/20201216/3.sorted_mask',
        out_root = '/data/medical/cardiac/cta2mbf/20201216/5.mbf_myocardium',
        mask_pattern = 'CTA/CTA_MASK_connected.nii.gz',
        registered_pattern = 'cta_mip_bf.nii.gz',
        out_mbf_pattern = 'mbf.nii.gz',
        myocardium_label = 6
    ):
    # mbf_root = '/data/medical/cardiac/cta2mbf/20201216/4.registration_batch1'
    # mask_root = '/data/medical/cardiac/cta2mbf/20201216/3.sorted_mask'
    # out_root = '/data/medical/cardiac/cta2mbf/20201216/5.mbf_myocardium'

    # mask_pattern = 'CTA/CTA_MASK_connected.nii.gz'
    # registered_pattern = 'cta_mip_bf.nii.gz'

    # out_mbf_pattern = 'mbf.nii.gz'

    # myocardium_label = 6

    pids = os.listdir(mbf_root)

    for pid in tqdm(pids):
        try:
            registered_mbf_file = os.path.join(mbf_root, pid, registered_pattern)
            mask_file = os.path.join(mask_root, pid, mask_pattern)
            
            # cta_file1 = os.path.join(mask_root, pid, 'CTA/CTA.nii.gz')
            # cta_file2 = os.path.join(mbf_root, pid, 'cta.nii.gz')
            
            if not os.path.isfile(registered_mbf_file):
                continue
            if not os.path.isfile(mask_file):
                continue
            
            # cta_image1 = DataIO.load_nii_image(cta_file1)['sitk_image']
            # cta_image2 = DataIO.load_nii_image(cta_file2)['sitk_image']

            mbf_image = DataIO.load_nii_image(registered_mbf_file)['sitk_image']
            mbf_mask = DataIO.load_nii_image(mask_file)['sitk_image']
            extracted_mbf_image = ImagePostProcessingUtils.extract_region_by_mask(mbf_image, mbf_mask, default_value=-1024, mask_label=myocardium_label)

            # 将左心室壁中小于0的值,设置为-1024，避免造成干扰
            tmp_arr = sitk.GetArrayFromImage(extracted_mbf_image)
            tmp_arr[tmp_arr<0] = -1024
            extracted_mbf_image = sitk.GetImageFromArray(tmp_arr)
            extracted_mbf_image.CopyInformation(mbf_image)

            out_sub_dir = os.path.join(out_root, pid)
            os.makedirs(out_sub_dir, exist_ok=True)
            out_mbf_file = os.path.join(out_sub_dir, out_mbf_pattern)

            sitk.WriteImage(extracted_mbf_image, out_mbf_file)
        except Exception as e:
            print(e)
            print('====> Error case:\t{}'.format(pid))
            

def step_5_2_extract_pericardium_bbox(
        mask_root = '/data/medical/cardiac/cta2mbf/20201216/3.sorted_mask',
        cta_root = '/data/medical/cardiac/cta2mbf/20201216/3.sorted_mask',
        mbf_root = '/data/medical/cardiac/cta2mbf/20201216/5.mbf_myocardium',
        out_root = '/data/medical/cardiac/cta2mbf/20201216/5.mbf_myocardium',
        mask_pattern = 'CTA/CTA_MASK_connected.nii.gz', 
        mbf_pattern = 'mbf.nii.gz'
    ):
    # mask_root = '/data/medical/cardiac/cta2mbf/20201216/3.sorted_mask'
    cta_root = mask_root
    # mbf_root = '/data/medical/cardiac/cta2mbf/20201216/5.mbf_myocardium'
    # out_root = '/data/medical/cardiac/cta2mbf/20201216/5.mbf_myocardium'
    # mask_pattern = 'CTA/CTA_MASK_connected.nii.gz'

    pids = os.listdir(mbf_root)

    for pid in tqdm(pids):
        cta_file = os.path.join(cta_root, pid, 'CTA/CTA.nii.gz')
        mbf_file = os.path.join(mbf_root, pid, mbf_pattern)
        mask_file = os.path.join(mask_root, pid, mask_pattern)
        
        mask_data = DataIO.load_nii_image(mask_file)
        cta_data = DataIO.load_nii_image(cta_file)
        mbf_data = DataIO.load_nii_image(mbf_file)

        mask_arr = mask_data['image']
        cta_arr = cta_data['image']
        mbf_arr = mbf_data['image']

        boundary_info = MaskBoundingUtils.extract_mask_arr_bounding(mask_arr)

        [min_z, min_y, min_x, max_z, max_y, max_x] = boundary_info
        
        cropped_cta_arr = cta_arr[min_z:max_z+1, min_y:max_y+1, min_x:max_x+1]
        cropped_mbf_arr = mbf_arr[min_z:max_z+1, min_y:max_y+1, min_x:max_x+1]
        cropped_mask_arr = mask_arr[min_z:max_z+1, min_y:max_y+1, min_x:max_x+1]

        cropped_cta_image = sitk.GetImageFromArray(cropped_cta_arr)
        cropped_cta_image.SetOrigin(cta_data['origin'])
        cropped_cta_image.SetSpacing(cta_data['spacing'])
        cropped_cta_image.SetDirection(cta_data['direction'])

        cropped_mbf_image = sitk.GetImageFromArray(cropped_mbf_arr)
        cropped_mbf_image.CopyInformation(cropped_cta_image)
        cropped_mask_image = sitk.GetImageFromArray(cropped_mask_arr)
        cropped_mask_image.CopyInformation(cropped_cta_image)

        out_sub_dir = os.path.join(out_root, pid)
        os.makedirs(out_sub_dir, exist_ok=True)
        
        out_cta_file = os.path.join(out_sub_dir, 'cropped_cta.nii.gz')
        out_mbf_file = os.path.join(out_sub_dir, 'cropped_mbf.nii.gz')
        out_mask_file = os.path.join(out_sub_dir, 'cropped_mask.nii.gz')

        sitk.WriteImage(cropped_cta_image, out_cta_file)
        sitk.WriteImage(cropped_mbf_image, out_mbf_file)
        sitk.WriteImage(cropped_mask_image, out_mask_file)

def analyze_data_cropped_cta(root_dir, cta_pattern='cropped_cta.nii.gz'):
    infos = []
    max_x = 0
    max_y = 0
    max_z = 0
    for pid in tqdm(os.listdir(root_dir)):
        cta_file = os.path.join(root_dir, '{}'.format(pid), cta_pattern)
        if not os.path.join(cta_file):
            print('{} not exist!'.format(cta_file))
            continue
        print(cta_file)
        image = sitk.ReadImage(cta_file)
        info = 'image shape:\t{}'.format(image.GetSize())
        infos.append(info)
        x,y,z = image.GetSize()
        if max_x < x:
            max_x = x
        if max_y < y:
            max_y = y
        if max_z < z:
            max_z = z
    print('max x:\t{}'.format(max_x))
    print('max y:\t{}'.format(max_y))
    print('max z:\t{}'.format(max_z))

    print('\n'.join(infos))

def extract_mbf_mask_onecase(sub_root_dir):
    mbf_file = os.path.join(sub_root_dir, 'cropped_mbf.nii.gz')
    mbf_mask_file = os.path.join(sub_root_dir, 'cropped_mbf_mask.nii.gz')
    mbf_img = sitk.ReadImage(mbf_file)
    mbf_arr = sitk.GetArrayFromImage(mbf_img)
    mbf_mask_arr = np.array(mbf_arr > 0, dtype=np.uint8)
    mbf_mask_img = sitk.GetImageFromArray(mbf_mask_arr)
    mbf_mask_img.CopyInformation(mbf_img)
    sitk.WriteImage(mbf_mask_img, mbf_mask_file)

def extract_mbf_mask(root_dir):
    for suid in tqdm(os.listdir(root_dir)):
        sub_root_dir = os.path.join(root_dir, '{}'.format(suid))
        extract_mbf_mask_onecase(sub_root_dir)



def preprocess_data_114_extract_modalitys(
        in_root, 
        out_root, 
        config_file = '/data/medical/cardiac/cta2mbf/data_114_20210715/annotation/00回流数据信息表格（UID)-DD1012(终稿).xlsx'
    ):
    # config_file = '/data/medical/cardiac/cta2mbf/data_114_20210318/annotation/01回流数据信息表格（UID)-DD1012(终稿).xlsx'
    # config_file = '/data/medical/cardiac/cta2mbf/data_66_20210517/annotation/01回流数据信息表格（UID)-DD1012(终稿).xlsx'
    # 老表格sheet_num为0
    # sheet_num = 0
    sheet_num = 1
    df = pd.read_excel(config_file, sheet_name = sheet_num, header = [1])
    print(df.columns)
    
    os.makedirs(out_root, exist_ok=True)

    for index, row in tqdm(df.iterrows()):
        pid = row['病人编号（Patient ID）']
        cta_suid = row['CTA']
        mip_suid = row['MIP']
        avg_suid = row['AVG']
        bf_suid = row['BF']

        src_cta_file = os.path.join(in_root, str(pid), cta_suid)
        dst_cta_file = os.path.join(out_root, str(pid), 'CTA', cta_suid)

        src_mip_file = os.path.join(in_root, str(pid), mip_suid)
        dst_mip_file = os.path.join(out_root, str(pid), 'MIP', mip_suid)

        src_avg_file = os.path.join(in_root, str(pid), avg_suid)
        dst_avg_file = os.path.join(out_root, str(pid), 'AVG', avg_suid)

        src_bf_file = os.path.join(in_root, str(pid), bf_suid)
        dst_bf_file = os.path.join(out_root, str(pid), 'BF', bf_suid)

        cta_exist = True
        mip_exist = True
        avg_exist = True
        bf_exist = True

        if not os.path.isdir(src_cta_file):
            cta_exist = False
            print('====>  CTA: No such file or directory:\t{}'.format(src_cta_file))
        if not os.path.isdir(src_mip_file):
            mip_exist = False
            print('====>  MIP: No such file or directory:\t{}'.format(src_mip_file))
        if not os.path.isdir(src_avg_file):
            avg_exist = False
            print('====>  AVG: No such file or directory:\t{}'.format(src_avg_file))
        if not os.path.isdir(src_bf_file):
            bf_exist = False
            print('====>  BF: No such file or directory:\t{}'.format(src_bf_file))

        if not (cta_exist and mip_exist and avg_exist and bf_exist):
            continue

        try:
            shutil.copytree(src_cta_file, dst_cta_file)
            shutil.copytree(src_mip_file, dst_mip_file)
            shutil.copytree(src_avg_file, dst_avg_file)
            shutil.copytree(src_bf_file, dst_bf_file)
        except Exception as e:
            print(e)
            print('====> Error case:\t', pid)

def preprocess_data_114():
    # root = '/data/medical/cardiac/cta2mbf/data_114_20210318'
    root = '/data/medical/cardiac/cta2mbf/data_114_20210715'
    
    # # step 2
    # in_root = os.path.join(root, '0.ori_2')
    # out_root = os.path.join(root, '0.ori_3')
    # sort_by_series_uid_multiprocessing(in_root, out_root, 24)

    # # step 3
    # in_root = os.path.join(root, '0.ori_3')
    # out_root = os.path.join(root, '3.sorted_dcm')
    # preprocess_data_114_extract_modalitys(in_root, out_root)

    # # step 3
    # in_root = os.path.join(root, '3.sorted_dcm')
    # out_root = os.path.join(root, '3.sorted_nii')
    # step_3_4_convert_dicom_series_to_nii(in_root, out_root)

    # step 4 generate registration cmd 
    '''
    配准过程执行data_preprocessing_registration.py
    '''
    # data_root = os.path.join(root, '3.sorted_nii')
    # data_root = '/data/zhangwd/data/cardiac/data_114/3.sorted_nii'
    # out_file = '/data/zhangwd/data/cardiac/data_114/3.sorted_nii_config/registration_mbf2cta_mip.sh'
    # elastic_path = '/data/hjy/elastix-5.0/elastix'
    # param1_path = '/data/hjy/elastix-5.0/params_copd/parameters_Affine.txt'
    # param2_path = '/data/hjy/elastix-5.0/params_copd/parameters_BSpline.txt'
    # param3_path = '/data/hjy/elastix-5.0/params_copd/parameters_Rigid.txt'

    # generate_register_cmd(data_root, out_file, elastic_path, param1_path, param2_path, param3_path)


    # step 5 chamber segmentation
    # data_root = os.path.join(root, '3.sorted_dcm')
    # out_dir = os.path.join(root, '3.sorted_mask')
    # cardiac_segmentation(data_root, out_dir)
    # step_3_3_segment_cardiac_connected_region(root_dir = os.path.join(root, '3.sorted_mask'))
    # '''
    # ====> Error case:       4383280
    # ====> Error case:       2909626
    # '''

    # step 6 extract myocardium from bf images
    # registration_root = '/data/zhangwd/data/cardiac/data_114/3.sorted_nii'
    # registration_root = '/data/medical/cardiac/cta2mbf/data_114_20210318/4.registration_batch'
    # mbf_root = registration_root
    # mask_root = os.path.join(root,'3.sorted_mask')
    # out_root = os.path.join(root,'5.mbf_myocardium')
    # mask_pattern = 'CTA/CTA_MASK_connected.nii.gz'
    # # registered_pattern = 'registration_bf_avg.mha'
    # registered_pattern = 'cta_mip_bf.nii.gz'
    # # out_mbf_pattern = 'registration_bf_avg_myocardium.nii.gz'
    # out_mbf_pattern = 'registration_cta_mip_bf_myocardium.nii.gz'
    # myocardium_label = 6
    # step_5_1_extract_mbf_myocardium(mbf_root, mask_root, out_root, mask_pattern, registered_pattern, out_mbf_pattern, myocardium_label)

    # step 7 extract bbox from cta images
    # mask_root = os.path.join(root, '3.sorted_mask')
    # cta_root = mask_root
    # mbf_root = os.path.join(root, '5.mbf_myocardium')
    # out_root = os.path.join(root, '5.mbf_myocardium')
    # mask_pattern = 'CTA/CTA_MASK_connected.nii.gz'
    # # mbf_pattern = 'registration_bf_avg_myocardium.nii.gz'
    # mbf_pattern = 'registration_cta_mip_bf_myocardium.nii.gz'
    # step_5_2_extract_pericardium_bbox(mask_root, cta_root, mbf_root, out_root, mask_pattern, mbf_pattern)

    # step 8 分析现有cta数据，心脏部分的size
    # cta_root = os.path.join(root, '5.mbf_myocardium')
    # analyze_data_cropped_cta(cta_root)

    # step 9
    cta_root = os.path.join(root, '5.mbf_myocardium')
    extract_mbf_mask(cta_root)

def preprocess_data_66():
    root = '/data/medical/cardiac/cta2mbf/data_66_20210517'
    
    # step 2
    # in_root = os.path.join(root, 'CTP灌注各类数据（李主任）')
    # out_root = os.path.join(root, '0.ori_3')
    # sort_by_series_uid_multiprocessing(in_root, out_root, 24)    

    # # step 3
    # in_root = os.path.join(root, '0.ori_3')
    # out_root = os.path.join(root, '3.sorted_dcm')
    # preprocess_data_114_extract_modalitys(in_root, out_root)

    # # step 3
    # in_root = os.path.join(root, '3.sorted_dcm')
    # out_root = os.path.join(root, '3.sorted_nii')
    # step_3_4_convert_dicom_series_to_nii(in_root, out_root)

    # step 4 generate registration cmd 
    '''
    配准过程执行data_preprocessing_registration.py
    '''

    # # step 5 chamber segmentation
    # data_root = os.path.join(root, '3.sorted_dcm')
    # out_dir = os.path.join(root, '3.sorted_mask')
    # cardiac_segmentation(data_root, out_dir)
    # step_3_3_segment_cardiac_connected_region(root_dir = os.path.join(root, '3.sorted_mask'))

    # step 6 extract myocardium from bf images
    # registration_root = os.path.join(root, '4.registration_batch')
    # mbf_root = registration_root
    # mask_root = os.path.join(root,'3.sorted_mask')
    # out_root = os.path.join(root,'5.mbf_myocardium')
    # mask_pattern = 'CTA/CTA_MASK_connected.nii.gz'
    # # registered_pattern = 'registration_bf_avg.mha'
    # registered_pattern = 'cta_mip_bf.nii.gz'
    # # out_mbf_pattern = 'registration_bf_avg_myocardium.nii.gz'
    # out_mbf_pattern = 'registration_cta_mip_bf_myocardium.nii.gz'
    # myocardium_label = 6
    # step_5_1_extract_mbf_myocardium(mbf_root, mask_root, out_root, mask_pattern, registered_pattern, out_mbf_pattern, myocardium_label)

    # step 7 extract bbox from cta images
    mask_root = os.path.join(root, '3.sorted_mask')
    cta_root = mask_root
    mbf_root = os.path.join(root, '5.mbf_myocardium')
    out_root = os.path.join(root, '5.mbf_myocardium')
    mask_pattern = 'CTA/CTA_MASK_connected.nii.gz'
    # mbf_pattern = 'registration_bf_avg_myocardium.nii.gz'
    mbf_pattern = 'registration_cta_mip_bf_myocardium.nii.gz'
    step_5_2_extract_pericardium_bbox(mask_root, cta_root, mbf_root, out_root, mask_pattern, mbf_pattern)

def preprocess_data_140():
    root = '/data/medical/cardiac/cta2mbf/data_140_20210602'
    
    # # step 2
    # in_root = os.path.join(root, 'CTP灌注各类数据（李主任）')
    # out_root = os.path.join(root, '0.ori_3')
    # sort_by_series_uid_multiprocessing(in_root, out_root, 24)    

    # # step 3
    # in_root = os.path.join(root, '0.ori_3')
    # out_root = os.path.join(root, '3.sorted_dcm')
    # preprocess_data_114_extract_modalitys(in_root, out_root)

    # # step 3
    # in_root = os.path.join(root, '3.sorted_dcm')
    # out_root = os.path.join(root, '3.sorted_nii')
    # step_3_4_convert_dicom_series_to_nii(in_root, out_root)
    
    # step 4 generate registration cmd 
    '''
    配准过程执行data_preprocessing_registration.py
    '''

    # # step 5 chamber segmentation
    # data_root = os.path.join(root, '3.sorted_dcm')
    # out_dir = os.path.join(root, '3.sorted_mask')
    # cardiac_segmentation_new_algo(data_root, out_dir)
    # step_3_3_segment_cardiac_connected_region(root_dir = os.path.join(root, '3.sorted_mask'))

    # step 6 extract myocardium from bf images
    registration_root = os.path.join(root, '4.registration_batch')
    mbf_root = registration_root
    mask_root = os.path.join(root,'3.sorted_mask')
    out_root = os.path.join(root,'5.mbf_myocardium')
    mask_pattern = 'CTA/CTA_MASK_connected.nii.gz'
    # registered_pattern = 'registration_bf_avg.mha'
    registered_pattern = 'cta_mip_bf.nii.gz'
    # out_mbf_pattern = 'registration_bf_avg_myocardium.nii.gz'
    out_mbf_pattern = 'registration_cta_mip_bf_myocardium.nii.gz'
    myocardium_label = 6
    step_5_1_extract_mbf_myocardium(mbf_root, mask_root, out_root, mask_pattern, registered_pattern, out_mbf_pattern, myocardium_label)

    # step 7 extract bbox from cta images
    mask_root = os.path.join(root, '3.sorted_mask')
    cta_root = mask_root
    mbf_root = os.path.join(root, '5.mbf_myocardium')
    out_root = os.path.join(root, '5.mbf_myocardium')
    mask_pattern = 'CTA/CTA_MASK_connected.nii.gz'
    # mbf_pattern = 'registration_bf_avg_myocardium.nii.gz'
    mbf_pattern = 'registration_cta_mip_bf_myocardium.nii.gz'
    step_5_2_extract_pericardium_bbox(mask_root, cta_root, mbf_root, out_root, mask_pattern, mbf_pattern)

    # step 8 分析现有cta数据，心脏部分的size
    # cta_root = os.path.join(root, '5.mbf_myocardium')
    # analyze_data_cropped_cta(cta_root)

    # step 9
    cta_root = os.path.join(root, '5.mbf_myocardium')
    extract_mbf_mask(cta_root)

# copy自监督可以使用的数据
def copy_ssl_data(data_root, out_root):
    os.makedirs(out_root, exist_ok=True)
    for suid in tqdm(os.listdir(data_root)):
        src_mbf_file = os.path.join(data_root, '{}'.format(suid), 'cropped_mbf.nii.gz')
        dst_mbf_file = os.path.join(out_root, '{}_cropped_mbf.nii.gz'.format(suid))
        shutil.copyfile(src_mbf_file, dst_mbf_file)


'''
这是个分界线，以下在扯淡
'''

'''
这里插入的是个题外话，数据入组的时候整错了，我能怎么办呢？
'''
def extract_cta_systole(root_dir):
    systold_pids = []
    mbf_root = os.path.join(os.path.dirname(root_dir), '5.mbf_myocardium')
    exist_pids = os.listdir(mbf_root)
    exist_pids = []
    for pid in tqdm(os.listdir(root_dir)):
        try:
            cta_root = os.path.join(root_dir, pid, 'CTA')
            cta_suid  = os.path.join(cta_root, os.listdir(cta_root)[0])
            metadata = DicomTagUtils.load_metadata(cta_suid, is_series=True)
            dsc = metadata.SeriesDescription
            if 'BestSyst' in dsc:
                systold_pids.append(pid)
            # print('hello world!')
        except:
            pass
    systold_pids = [os.path.join(mbf_root, i) for i in systold_pids if i in exist_pids]
    return systold_pids


def extract_cta_systole_batch(out_dir='/data/medical/cardiac/cta2mbf/data_155_20210628/5.mbf_myocardium'):
    systold_pids = []
    # tmp_pids = extract_cta_systole('/data/medical/cardiac/cta2mbf/data_114_20210318/3.sorted_dcm')
    tmp_pids = extract_cta_systole('/data/medical/cardiac/cta2mbf/data_114_20210715/3.sorted_dcm')
    systold_pids += tmp_pids
    # tmp_pids = extract_cta_systole('/data/medical/cardiac/cta2mbf/data_140_20210602/3.sorted_dcm')    
    # systold_pids += tmp_pids
    print(systold_pids)

    # os.makedirs(out_dir, exist_ok=True)
    # exist_ids = []
    # for pid in tqdm(systold_pids):
    #     try:
    #         src = pid
    #         num = os.path.basename(src)
    #         if num in exist_ids:
    #             continue
    #         dst = os.path.join(out_dir, num)
    #         shutil.copytree(src, dst)
    #         exist_ids.append(num)
    #     except:
    #         pass

def preprocess_data_140_update():
    root = '/data/medical/cardiac/cta2mbf/data_140_20210602_update'
    # # step 2
    # in_root = os.path.join(root, 'data_wtf_127_20210702')
    # out_root = os.path.join(root, '0.ori_3')
    # sort_by_series_uid_multiprocessing(in_root, out_root, 24)    

    # # step 3
    in_root = os.path.join(root, '0.ori_3')
    out_root = os.path.join(root, '3.sorted_dcm')
    for pid in tqdm(os.listdir(in_root)):
        pid_path = os.path.join(in_root, pid)
        src_files = os.listdir(pid_path)
        if len(src_files) != 1:
            print('updated CTA files error:\t{}'.format(pid))
            continue
        src_file = os.path.join(pid_path, src_files[0])
        dst_root = os.path.join(out_root, pid, 'CTA')
        shutil.rmtree(dst_root)
        os.makedirs(dst_root, exist_ok=True)
        dst_file = os.path.join(dst_root, src_files[0])
        shutil.copytree(src_file, dst_file)
    
    # preprocess_data_114_extract_modalitys(in_root, out_root)

    '''
    剩下的将文件夹名字修改为“data_140_20210602”，并继续preprocess_data_140()的代码
    '''


if __name__ == '__main__':
    # step_1_crop_from_ori1_to_ori2()
    # step_2_sort_by_series_uid()
    # test_extract_ctp_related()
    # step_3_1_extract_ctp_related()
    # step_3_2_extract_cta_file()
    # step_3_3_segment_cardiac()
    # step_3_4_convert_dicom_series_to_nii()
    # step_5_1_extract_mbf_myocardium()
    # step_5_2_extract_pericardium_bbox()
    # generate_register_cmd()
    # test_register_images()
    # check_last_5()

    # preprocess_data_114_extract_modalitys(None, None)
    # preprocess_data_114()
    # preprocess_data_66()
    # preprocess_data_140()
    
    # # 生成自监督数据
    # copy_ssl_data(
    #         '/data/medical/cardiac/cta2mbf/data_114_20210318/5.mbf_myocardium', 
    #         '/data/medical/cardiac/cta2mbf/ssl/cropped_ori'
    #     )
    # copy_ssl_data(
    #         '/data/medical/cardiac/cta2mbf/data_140_20210602/5.mbf_myocardium', 
    #         '/data/medical/cardiac/cta2mbf/ssl/cropped_ori'
    #     )

    extract_cta_systole_batch()

    # preprocess_data_140_update()