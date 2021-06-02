import os
import sys
from glob import glob
import numpy as np
import pydicom

import SimpleITK as sitk

# 提取dicom序列的序列号
def extract_series_uid(dicom_series_path):
    '''
    dicom_series_path: '../../../data/sxy_data/data_category/NEWCTA/1315171/DS_CorAdSeq  0.75  Bv40  3  BestSyst 44 %'
    debug cmd: extract_series_uid('../../../data/sxy_data/data_category/NEWCTA/1315171/DS_CorAdSeq  0.75  Bv40  3  BestSyst 44 %')
    '''
    series_reader = sitk.ImageSeriesReader()
    dicomfilenames = series_reader.GetGDCMSeriesFileNames(dicom_series_path)
    series_reader.SetFileNames(dicomfilenames)
    
    metadata = pydicom.dcmread(dicomfilenames[0])

    try:
        series_uid = metadata.SeriesInstanceUID
    except:
        pass

    return series_uid

def extract_series_uids(dicom_series_root):
    '''
    debug cmd: extract_series_uids('../../../data/sxy_data/data_category/NEWCTA')
    '''
    for pid in os.listdir(dicom_series_root):
        patient_path = os.path.join(dicom_series_root, pid)
        if not os.path.isdir(patient_path):
            continue
        sub_paths = os.listdir(patient_path)
        dicom_series_path = None
        for sub_path in sub_paths:
            if 'DS_CorAdSeq  0.75' in sub_path:
                dicom_series_path = sub_path
        dicom_series_path = os.path.join(patient_path, dicom_series_path)
        print(extract_series_uid(dicom_series_path))


if __name__ == '__main__':
    # extract_series_uid('../../../data/sxy_data/data_category/NEWCTA/1315171/DS_CorAdSeq  0.75  Bv40  3  BestSyst 44 %')
    extract_series_uids('../../../data/sxy_data/data_category/NEWCTA')