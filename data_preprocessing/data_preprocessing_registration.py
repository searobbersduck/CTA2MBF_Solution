'''
Description: registration cta/ctp mip/ctp avg/ctp bf images with elastix library.
Version: 1.0
Autor: searobbersanduck
Date: 2021-01-06 09:51:38
LastEditors: searobbersanduck
LastEditTime: 2021-06-02 10:56:43
License : (C)Copyright 2020-2021, MIT
'''

'''
由于环境安装问题，目前该代码需切换到elastix安装路径下执行。
cp data_preprocessing_registration.py /home/zhangwd/code/pkg/SimpleElastix/build/SimpleITK-build/Wrapping/Python/
cd /home/zhangwd/code/pkg/SimpleElastix/build/SimpleITK-build/Wrapping/Python/

source activate pytorch1.6
python data_preprocessing_registration.py
'''

import os
import sys
from glob import glob
from tqdm import tqdm
import time
import numpy as np

import SimpleITK as sitk

# 情况特殊，此处写绝对路径
sys.path.append('/home/zhangwd/code/work')
from MedCommon.utils.data_io_utils import DataIO

selx = sitk.ElastixImageFilter()
print('\n'.join(dir(selx)))


def elastix_register_images_one_case(cta_file, mip_file, avg_file, bf_file, out_dir, is_dcm=False):
    if is_dcm:
        cta_data = DataIO.load_dicom_series(cta_file)
        mip_data = DataIO.load_dicom_series(mip_file)
        avg_data = DataIO.load_dicom_series(avg_file)
        bf_data = DataIO.load_dicom_series(bf_file)
    else:
        cta_data = DataIO.load_nii_image(cta_file)
        mip_data = DataIO.load_nii_image(mip_file)
        avg_data = DataIO.load_nii_image(avg_file)
        bf_data = DataIO.load_nii_image(bf_file)
    cta_img = cta_data['sitk_image']
    mip_img = mip_data['sitk_image']
    avg_img = avg_data['sitk_image']
    bf_img = bf_data['sitk_image']

    selx = sitk.ElastixImageFilter()
    selx.SetFixedImage(cta_img)
    selx.SetMovingImage(mip_img)
    selx.SetParameterMap(selx.GetDefaultParameterMap('nonrigid'))
    selx.Execute()

    moved_mip_img_according_mip_trans = sitk.Transformix(mip_img, selx.GetTransformParameterMap())
    moved_mip_img_according_mip_trans.CopyInformation(cta_img)
    moved_avg_img_according_mip_trans = sitk.Transformix(avg_img, selx.GetTransformParameterMap())
    moved_avg_img_according_mip_trans.CopyInformation(cta_img)
    moved_bf_img_according_mip_trans = sitk.Transformix(bf_img, selx.GetTransformParameterMap())
    moved_bf_img_according_mip_trans.CopyInformation(cta_img)

    out_cta_mip_mip_file = os.path.join(out_dir, 'cta_mip_mip.nii.gz')
    out_cta_avg_mip_file = os.path.join(out_dir, 'cta_mip_avg.nii.gz')
    out_cta_bf_mip_file = os.path.join(out_dir, 'cta_mip_bf.nii.gz')

    out_cta_mip_avg_file = os.path.join(out_dir, 'cta_avg_mip.nii.gz')
    out_cta_avg_avg_file = os.path.join(out_dir, 'cta_avg_avg.nii.gz')
    out_cta_bf_avg_file = os.path.join(out_dir, 'cta_avg_bf.nii.gz')

    os.makedirs(out_dir, exist_ok=True)
    out_cta_file = os.path.join(out_dir, 'cta.nii.gz')

    print('{}:\t{}'.format(out_cta_mip_mip_file, moved_mip_img_according_mip_trans.GetSize()))
    print('{}:\t{}'.format(out_cta_avg_mip_file, moved_avg_img_according_mip_trans.GetSize()))
    print('{}:\t{}'.format(out_cta_bf_mip_file, moved_bf_img_according_mip_trans.GetSize()))

    sitk.WriteImage(moved_mip_img_according_mip_trans, out_cta_mip_mip_file)
    sitk.WriteImage(moved_avg_img_according_mip_trans, out_cta_avg_mip_file)
    sitk.WriteImage(moved_bf_img_according_mip_trans, out_cta_bf_mip_file)

    sitk.WriteImage(cta_img, out_cta_file)

def test_elastix_register_images_one_case():
    pid = '1315171'
    data_root = '/data/medical/cardiac/cta2mbf/20201216/3.sorted/'
    pid_path = os.path.join(data_root, pid)
    cta_dir = os.path.join(pid_path, 'CTA')
    cta_files = os.listdir(cta_dir)
    cta_file = os.path.join(cta_dir, cta_files[0])
    mip_dir = os.path.join(pid_path, 'MIP')
    mip_files = os.listdir(mip_dir)
    mip_file = os.path.join(mip_dir, mip_files[0])

    avg_dir = os.path.join(pid_path, 'AVG')
    avg_files = os.listdir(avg_dir)
    avg_file = os.path.join(avg_dir, avg_files[0])

    bf_dir = os.path.join(pid_path, 'BF')
    bf_files = os.listdir(bf_dir)
    bf_file = os.path.join(bf_dir, bf_files[0])
    
    print(cta_file)
    print(mip_file)
    print(avg_file)
    print(bf_file)

    out_dir = '/data/medical/cardiac/cta2mbf/20201216/4.registration_test/{}'.format(pid)

    beg = time.time()
    # register_images(cta_file, mip_file, bf_file, True)
    elastix_register_images_one_case(cta_file, mip_file, avg_file, bf_file, out_dir, True)
    end = time.time()
    print('====> test_register_images time cosume is:\t{:.3f}'.format(end-beg))
    
def elastix_register_images_single_task(data_root, out_dir, pids, task_id):
    log = []
    for pid in tqdm(pids):
        try:
            pid_path = os.path.join(data_root, pid)
            cta_dir = os.path.join(pid_path, 'CTA')
            cta_files = os.listdir(cta_dir)
            cta_file = os.path.join(cta_dir, cta_files[0])
            mip_dir = os.path.join(pid_path, 'MIP')
            mip_files = os.listdir(mip_dir)
            mip_file = os.path.join(mip_dir, mip_files[0])

            avg_dir = os.path.join(pid_path, 'AVG')
            avg_files = os.listdir(avg_dir)
            avg_file = os.path.join(avg_dir, avg_files[0])

            bf_dir = os.path.join(pid_path, 'BF')
            bf_files = os.listdir(bf_dir)
            bf_file = os.path.join(bf_dir, bf_files[0])

            out_pid_dir = os.path.join(out_dir, pid)

            elastix_register_images_one_case(cta_file, mip_file, avg_file, bf_file, out_pid_dir, True)
            
            
        except Error as e:
            print(e)
            print('====> Error case:\t{}'.format(pid))
            log.append(e)
            log.append('====> Error case:\t{}'.format(pid))

    with open('log_{}'.format(task_id), 'w') as f:
        f.write('\n'.join(log))

def elastix_register_images_multi_task(data_root, out_dir, process_num=6, reuse=False):
    pids = []
    gen_pids = []
    for pid in os.listdir(out_dir):
        gen_pids.append(pid)

    for pid in os.listdir(data_root):
        if len(pid) == 7:
            if reuse:
                if pid not in gen_pids:
                    pids.append(pid)
            else:
                pids.append(pid)
    
    num_per_process = (len(pids) + process_num - 1)//process_num

    # this for single thread to debug
    # elastix_register_images_single_task(data_root, out_dir, pids)

    # this for run 
    import multiprocessing
    from multiprocessing import Process
    multiprocessing.freeze_support()

    pool = multiprocessing.Pool()

    results = []

    print(len(pids))
    for i in range(process_num):
        sub_pids = pids[num_per_process*i:min(num_per_process*(i+1), len(pids))]
        print(len(sub_pids))
        result = pool.apply_async(elastix_register_images_single_task, 
            args=(data_root, out_dir, sub_pids, i))
        results.append(result)

    pool.close()
    pool.join()
    
def test_elastix_register_images_multi_task():
    # data_root = '/data/medical/cardiac/cta2mbf/20201216/3.sorted'
    # out_dir = '/data/medical/cardiac/cta2mbf/20201216/4.registration_batch1'
    data_root = '/data/medical/cardiac/cta2mbf/data_114_20210318/3.sorted_dcm'
    out_dir = '/data/medical/cardiac/cta2mbf/data_114_20210318/4.registration_batch'
    data_root = '/data/medical/cardiac/cta2mbf/data_66_20210517/3.sorted_dcm'
    out_dir = '/data/medical/cardiac/cta2mbf/data_66_20210517/4.registration_batch'
    data_root = '/data/medical/cardiac/cta2mbf/data_140_20210602/3.sorted_dcm'
    out_dir = '/data/medical/cardiac/cta2mbf/data_140_20210602/4.registration_batch'    
    os.makedirs(out_dir, exist_ok=True)    
    elastix_register_images_multi_task(data_root, out_dir, 6, reuse=True)


def step_4_register_images():
    test_elastix_register_images_multi_task()

    



if __name__ == '__main__':
    # test_elastix_register_images_one_case()
    # test_elastix_register_images_multi_task()
    step_4_register_images()
    # pass