import os
from glob import glob
import numpy as np
import SimpleITK as sitk
import time
import pydicom
import shutil
import nibabel as nib
import imageio
import csv


#读医院数据文件夹里的dicom，取病人编号一致和且series名称是ctp的复制到一个文件夹
def extract_from_hospital_folder(in_folder, out_folder, series_category):
    '''
    extract_from_hospital_folder('D:\sixth_hospital_20200616_CTP\DICOM\six_ctp', 'D:/CTP/ctp_only/'，'MIP')
    '''
    '''
    in_folder文件夹形式如下: 
    .
    ├── patient_id
    │   ├── slice1
    │   └── slice2
    │   ├── slice3
    │   └── ...   

    out_folder文件夹形式如下(如果是MIP/MBF): 
    .
    ├── patient_id
    │   ├── MBF_patient-id_dicom file原名
    │   └── MBF_patient-id_dicom file原名
    │   ├── MBF_patient-id_dicom file原名
    │   └── ...  

    out_folder文件夹形式如下(如果是CTA): 
    .
    └── patient_id
        └── CTA series_description (多个cta序列作为二级文件名)
            ├── CTA_patient-id_dicom file原名
            ├── CTA_patient-id_dicom file原名
            ├── ...

    '''
    patient_folders = os.listdir(in_folder)
    for patient_folder in patient_folders:
        patient_folder_path = os.path.join(in_folder, patient_folder)
        dcm_files =  os.listdir(patient_folder_path)
        for dcm_file in dcm_files:
            dcm_file_path = os.path.join(patient_folder_path,dcm_file)
            metadata = pydicom.dcmread(dcm_file_path)
            patient_id = metadata.PatientID
            series_description = metadata.SeriesDescription
            if series_category == 'MIP':
                if 'VPCT MIP #1' == series_description:
                    savedir = os.path.join(out_folder, patient_id)
                    os.makedirs(savedir, exist_ok=True)
                    save_path = os.path.join(savedir,  'MIP_'+ patient_id + '_' + os.path.basename(dcm_file))
                    shutil.copyfile(dcm_file_path, save_path)
            if series_category == 'MBF':      
                if 'VPCT BF #1' == series_description:
                    savedir = os.path.join(out_folder, patient_id)
                    os.makedirs(savedir, exist_ok=True)
                    save_path = os.path.join(savedir,  'MBF_'+ patient_id + '_' + os.path.basename(dcm_file))
                    shutil.copyfile(dcm_file_path, save_path)
            if series_category == 'CTA':
                if 'BestSyst' in series_description:
                    savedir = os.path.join(out_folder,patient_id,series_description)
                    os.makedirs(savedir, exist_ok=True)
                    save_path = os.path.join(savedir, 'CTA_' + patient_id + '_' + os.path.basename(dcm_file))
                    shutil.copyfile(dcm_file_path, save_path)
            
        print('ok')



#读文件夹里的dicom封装成一个对象 转换为nii.gz去配准和计算
def dcm_to_nii(in_dcm_path, out_nii_path,series_category):
    patient_folders = os.listdir(in_dcm_path)
    if series_category == 'CTA':
        for patient_folder in patient_folders:
            patient_folder_path = os.path.join(in_dcm_path, patient_folder)
            cta_folders = os.listdir(patient_folder_path) #每个病人下的cta
            for cta_folder in cta_folders:
                if 'DS_CorAdSeq  0.75  Bv40  3  BestSyst' in  cta_folder: #选择符合要求的CTA序列文件夹转nii.gz
                    cta_folder_path = os.path.join(patient_folder_path, cta_folder)
                    dcm_files =  os.listdir(cta_folder_path) 
                    file_num = len(dcm_files) # dcm文件数
                    print(file_num)

                    series_reader = sitk.ImageSeriesReader()
                    dicomfilenames = series_reader.GetGDCMSeriesFileNames(cta_folder_path)
                    series_reader.SetFileNames(dicomfilenames)

                    series_reader.MetaDataDictionaryArrayUpdateOn()
                    series_reader.LoadPrivateTagsOn()

                    image = series_reader.Execute()
                    dst_path = os.path.join(out_nii_path, patient_folder)
                    os.makedirs(dst_path, exist_ok=True) 
                    #nii的地址,命名为病人id_series_category_num_层数.nii.gz
                    dst = os.path.join(dst_path, patient_folder +'_' + series_category + '_num_'+ str(file_num) + '.nii.gz') #nii的地址,命名为病人__i_numxx.nii.gz
                    sitk.WriteImage(image, dst) #转换为nii.gz
            print('ok')

    else:    
        for patient_folder in patient_folders:
            patient_folder_path = os.path.join(in_dcm_path, patient_folder)
            dcm_files =  os.listdir(patient_folder_path) 
            file_num = len(dcm_files) # dcm文件数

            series_reader = sitk.ImageSeriesReader()
            dicomfilenames = series_reader.GetGDCMSeriesFileNames(patient_folder_path)
            series_reader.SetFileNames(dicomfilenames)

            series_reader.MetaDataDictionaryArrayUpdateOn()
            series_reader.LoadPrivateTagsOn()

            image = series_reader.Execute()
            dst_path = os.path.join(out_nii_path, patient_folder)
            os.makedirs(dst_path, exist_ok=True) 
            #nii的地址,命名为病人id_series_category_num_层数.nii.gz
            dst = os.path.join(dst_path, patient_folder +'_'+ series_category+ '_num_'+ str(file_num) + '.nii.gz') 
            sitk.WriteImage(image, dst)  
            print('ok')




#输出CTA和MBF的采集时间，看是否相近
def check_acq_time(cta_path,mbf_path):
    patient_folders = os.listdir(cta_path)
    for patient_folder in patient_folders:
        cta_folder_path = os.path.join(cta_path, patient_folder) 
        cta_series_folders = os.listdir(cta_folder_path)
        for cta_series in cta_series_folders:
            if 'DS_CorAdSeq  0.75  Bv40  3  BestSyst' in cta_series:
                cta_series_path = os.path.join(cta_folder_path,cta_series)
                dcm_files =  os.listdir(cta_series_path) 
                dcm_file_path = os.path.join(cta_series_path,dcm_files[0])
                metadata = pydicom.dcmread(dcm_file_path) 
                cta_acq_time = metadata.AcquisitionDateTime
        
        mbf_folder_path = os.path.join(mbf_path, patient_folder) 
        mbf_dcm_files =  os.listdir(mbf_folder_path) 
        mbf_dcm_file_path = os.path.join(mbf_folder_path,mbf_dcm_files[0])
        mbf_metadata = pydicom.dcmread(mbf_dcm_file_path) 
        mbf_acq_time = mbf_metadata.AcquisitionDateTime
        print('patient_id_'+patient_folder+'_cta_acq_time:'+cta_acq_time)
        print('patient_id_'+patient_folder+'_mbf_acq_time:'+mbf_acq_time)




#计数CTP文件夹下 每个子文件夹（病人）下 nii.gz的数量并输出
def count_num(in_folder):
    patient_folders = os.listdir(in_folder)
    for patient_folder in patient_folders:
        patient_folder_path = os.path.join(in_folder, patient_folder)

        ctp_folders = os.listdir(patient_folder_path) 
        ctp_folders.sort()
        folder_num = len(ctp_folders)
        for i in range(folder_num):
            ctp_folder_path = os.path.join(patient_folder_path, ctp_folders[i])
            dcm_files =  os.listdir(ctp_folder_path) #计数
            file_num = len(dcm_files)  
            print(patient_folder + ctp_folders[i]+'_number:'+str(file_num))  


def resample_sitkImage_by_spacing(sitkImage, newSpacing, vol_default_value='min', interpolator=sitk.sitkLinear):
    """
    :param sitkImage:
    :param newSpacing:
    :return:
    """
    if sitkImage == None:
        return None
    if newSpacing is None:
        return None

    dim = sitkImage.GetDimension()
    if len(newSpacing) != dim:
        return None

    # determine the default value
    vol_value = 0.0
    if vol_default_value == 'min':
        vol_value = float(np.ndarray.min(sitk.GetArrayFromImage(sitkImage)))
    elif vol_default_value == 'zero':
        vol_value = 0.0
    elif str(vol_default_value).isnumeric():
        vol_value = float(vol_default_value)

    # calculate new size
    np_oldSize = np.array(sitkImage.GetSize())
    np_oldSpacing = np.array(sitkImage.GetSpacing())
    print(np_oldSize)

    np_newSpacing = np.array(newSpacing)
    np_newSize = np.divide(np.multiply(np_oldSize, np_oldSpacing), np_newSpacing)
    newSize = tuple(np_newSize.astype(np.uint).tolist())
    print(np_newSize)

    # resample sitkImage into new specs
    transform = sitk.Transform()

    return sitk.Resample(sitkImage, newSize, transform, interpolator, sitkImage.GetOrigin(),
                         newSpacing, sitkImage.GetDirection(), vol_value, sitkImage.GetPixelID())

    
def resample(in_folder,out_folder, interpolator_method):
    patient_folders = os.listdir(in_folder)
    for patient_folder in patient_folders:
        patient_folder_path = os.path.join(in_folder, patient_folder)
        files = os.listdir(patient_folder_path)
        for nii_file in  files:
            if 'CTA' in nii_file:
                nii_path = os.path.join(patient_folder_path,nii_file)
                nii_image = sitk.ReadImage(nii_path, sitk.sitkFloat32)
                cta_oldspacing = np.array(nii_image.GetSpacing())
                print(cta_oldspacing)        
                cta_newspacing = np.array((cta_oldspacing[0],cta_oldspacing[0],cta_oldspacing[0]))
                print(cta_newspacing)
                resample_cta = resample_sitkImage_by_spacing(nii_image, cta_newspacing, vol_default_value='min', interpolator=interpolator_method)
                np_newsize = np.array(resample_cta.GetSize())
                dst_path = os.path.join(out_folder,patient_folder)
                os.makedirs(dst_path, exist_ok=True)
                dst = os.path.join(dst_path,patient_folder+'_bspline_resampled_CTA_'+str(np_newsize[0])+'_'+str(np_newsize[1])+'_'+str(np_newsize[2])+'.nii.gz')
                sitk.WriteImage(resample_cta,dst)
            if 'MIP' in nii_file:
                nii_path = os.path.join(patient_folder_path,nii_file)
                nii_image = sitk.ReadImage(nii_path, sitk.sitkFloat32)
                mip_oldspacing = np.array(nii_image.GetSpacing())
                print(mip_oldspacing)        
                mip_newspacing = cta_newspacing
                print(mip_newspacing)
                resample_mip = resample_sitkImage_by_spacing(nii_image, mip_newspacing, vol_default_value='min', interpolator=interpolator_method)
                np_newsize = np.array(resample_mip.GetSize())
                dst_path = os.path.join(out_folder,patient_folder)
                os.makedirs(dst_path, exist_ok=True)
                dst = os.path.join(dst_path,patient_folder+'_bspline_resampled_MIP_'+str(np_newsize[0])+'_'+str(np_newsize[1])+'_'+str(np_newsize[2])+'.nii.gz')
                sitk.WriteImage(resample_mip,dst)
            if 'MBF' in nii_file:
                nii_path = os.path.join(patient_folder_path,nii_file)
                nii_image = sitk.ReadImage(nii_path, sitk.sitkFloat32)
                mbf_oldspacing = np.array(nii_image.GetSpacing())
                print(mbf_oldspacing)        
                mbf_newspacing = cta_newspacing
                print(mbf_newspacing)
                resample_mbf = resample_sitkImage_by_spacing(nii_image, mbf_newspacing, vol_default_value='min', interpolator=interpolator_method)
                np_newsize = np.array(resample_mbf.GetSize())
                dst_path = os.path.join(out_folder,patient_folder)
                os.makedirs(dst_path, exist_ok=True)
                dst = os.path.join(dst_path,patient_folder+'_bspline_resampled_MBF_'+str(np_newsize[0])+'_'+str(np_newsize[1])+'_'+str(np_newsize[2])+'.nii.gz')
                sitk.WriteImage(resample_mbf,dst)

#将3d格式的nii切片为npy格式以供训练
def slicing(nii_path,npy_path,png_path):
    series_categories = os.listdir(nii_path)
    for series_category in series_categories:
        if 'CTA' in series_category:
            cta_path = os.path.join(nii_path,series_category)
            # cta = sitk.ReadImage(cta_path, sitk.sitkFloat32)
            cta = nib.load(cta_path).get_data() #载入(W, H, C)
            cta = np.array(cta)
            cta = cta.transpose((1, 0, 2))#(H, W, C)
            print(cta.shape)
            for i in range(90,231,1):
                slice_cta = cta[:,:,i]
                # slice_mbf = mbf[:,:,i]
                dst_cta_folder= os.path.join(npy_path,'cta')
                os.makedirs(dst_cta_folder, exist_ok=True) 
                dst_cta = os.path.join(dst_cta_folder,'3911806_cta_'+str(i)+'.npy')

                dst_cta_folder_png= os.path.join(png_path,'cta')
                os.makedirs(dst_cta_folder_png, exist_ok=True)         
                dst_cta_png = os.path.join(dst_cta_folder_png,'3911806_cta_'+str(i)+'.png')

                np.save(dst_cta,slice_cta)

                imageio.imwrite(dst_cta_png,slice_cta)

                print('ok')
        if 'MBF' in series_category:
            mbf_path = os.path.join(nii_path,series_category)
            # cta = sitk.ReadImage(cta_path, sitk.sitkFloat32)
            mbf = nib.load(mbf_path).get_data() #载入
            mbf = np.array(mbf)
            mbf = mbf.transpose((1, 0, 2))#(H, W, C)
            print(mbf.shape)
            for i in range(90,231,1):
                # slice_cta = cta[:,:,i]
                slice_mbf = mbf[:,:,i]


                dst_mbf_folder = os.path.join(npy_path,'mbf')
                os.makedirs(dst_mbf_folder, exist_ok=True) 
                dst_mbf = os.path.join(dst_mbf_folder,'3911806_mbf_'+str(i)+'.npy')

                dst_mbf_folder_png= os.path.join(png_path,'mbf')
                os.makedirs(dst_mbf_folder_png, exist_ok=True)         
                dst_mbf_png = os.path.join(dst_mbf_folder_png,'3911806_mbf_'+str(i)+'.png')

                np.save(dst_mbf,slice_mbf)

                imageio.imwrite(dst_mbf_png,slice_mbf)
                print('ok')


def copy_files_newfolder(ori_path, patient_folders,new_path):
    # patient_folders = os.listdir(ori_path)

    for patient_folder in patient_folders:
        patient_folder_path = os.path.join(ori_path,patient_folder)
        series_list = os.listdir(patient_folder_path)
        for series in series_list:
            series_path = os.path.join(patient_folder_path,series)
            file_list = os.listdir(series_path)
            for file in file_list:
                file_path = os.path.join(series_path,file)
                
                new_file_folder = os.path.join(new_path,patient_folder)
                os.makedirs(new_file_folder, exist_ok=True)
                new_file_path = os.path.join(new_file_folder,file)


                shutil.copyfile(file_path, new_file_path)
    print('ok')
        

# write the file name in training dataset and test dataset to a txt file
def write_config_txt(train_folder,val_folder,config_folder):
    train_patients = os.listdir(train_folder)
    cta_train_list = []
    mbf_train_list = []
    for train_patient in train_patients:
        train_patient_path = os.path.join(train_folder,train_patient)
        npyfiles = os.listdir(train_patient_path)
        for npyfile in npyfiles:
            if 'cta' in npyfile:
                cta_path  = os.path.join(train_patient,npyfile)
                cta_train_list.append(cta_path)
            if 'mbf' in npyfile:
                mbf_path  = os.path.join(train_patient,npyfile)
                mbf_train_list.append(mbf_path)
    cta_train_list.sort()
    mbf_train_list.sort()
    print(len(mbf_train_list))
    train_csv_path = os.path.join(config_folder,'config_2d_cta2mbf_train.txt')
    train_rows = zip(cta_train_list, mbf_train_list)
    with open(train_csv_path, "w") as f:
        writer = csv.writer(f,delimiter='\t')
        for train_row in train_rows:
            writer.writerow(train_row)      
    if val_folder:
        val_patients = os.listdir(val_folder)
        cta_val_list = []
        mbf_val_list = []
        for val_patient in val_patients:
            val_patient_path = os.path.join(val_folder,val_patient)
            val_npyfiles = os.listdir(val_patient_path)
            for val_npyfile in val_npyfiles:
                if 'cta' in val_npyfile:
                    cta_val_path  = os.path.join(val_patient,val_npyfile)
                    cta_val_list.append(cta_val_path)
                if 'mbf' in val_npyfile:
                    mbf_val_path  = os.path.join(val_patient,val_npyfile)
                    mbf_val_list.append(mbf_val_path)
        cta_val_list.sort()
        mbf_val_list.sort()
        print(len(mbf_val_list))
        val_csv_path = os.path.join(config_folder,'config_2d_cta2mbf_val.txt')
        val_rows = zip(cta_val_list, mbf_val_list)
        with open(val_csv_path, "w") as f2:
            writer = csv.writer(f2,delimiter='\t')
            for val_row in val_rows:
                writer.writerow(val_row) 

#把病人npy文件夹下的npy转为3d的nii.gz
def npy2nii(npy_path,nii_path):
    os.makedirs(nii_path,exist_ok=True)

    patient_folders = os.listdir(npy_path)
    for patient_folder in patient_folders:
        patient_folder_path = os.path.join(npy_path,patient_folder)
        series_categories = os.listdir(patient_folder_path)
        for series_category in series_categories:
            if 'cta' in series_category:
                cta_path = os.path.join(patient_folder_path,series_category)
                imgs = os.listdir(cta_path)
                imgs.sort(key=lambda x:int(x[12:-4]))
                print(imgs)
                img_3d = np.zeros((512,512,len(imgs)))
                for i in range(len(imgs)):
                    img_slice_path = os.path.join(cta_path,imgs[i])
                    img_slice = np.load(img_slice_path)
                    img_3d[:,:,i]=img_slice
                print(img_3d.shape)
                cta_img_nii= sitk.GetImageFromArray(img_3d)
                cta_dst = os.path.join(nii_path, patient_folder+'_'+'cta'+'_'+str(len(imgs)) +'.nii.gz')
                sitk.WriteImage(cta_img_nii,cta_dst)
            if 'mbf' in series_category:
                mbf_path = os.path.join(patient_folder_path,series_category)
                imgs = os.listdir(mbf_path)
                print(imgs)
                imgs.sort(key=lambda x:int(x[12:-4]))
                mbf_img_3d = np.zeros((512,512,len(imgs)))
                for i in range(len(imgs)):
                    img_slice_path = os.path.join(mbf_path,imgs[i])
                    img_slice = np.load(img_slice_path)
                    mbf_img_3d[:,:,i]=img_slice                
                    
                print(mbf_img_3d.shape)
                mbf_img_nii= sitk.GetImageFromArray(mbf_img_3d)
                mbf_dst = os.path.join(nii_path, patient_folder+'_'+'mbf'+'_'+str(len(imgs)) +'.nii.gz')
                sitk.WriteImage(mbf_img_nii,mbf_dst)


if __name__ =='__main__':
    # in_folder = ''
    # MIP_folder =''
    # CTA_folder = ''
    # MBF_folder = ''

    # extract_from_hospital_folder(in_folder, MIP_folder, 'MIP')
    # extract_from_hospital_folder(in_folder, MBF_folder, 'MBF')
    # extract_from_hospital_folder(in_folder, CTA_folder, 'CTA')
    # count_num(MBF_folder)
    # check_acq_time(CTA_folder,MBF_folder)
    # nii_path =  ''
    # dcm_to_nii(CTA_folder, nii_path,'CTA')
    # dcm_to_nii(MIP_folder, nii_path,'MIP')
    # dcm_to_nii(MBF_folder, nii_path,'MBF')
    # resample_nii_path = 'D:/resample_nii'
    # resample(nii_path, resample_nii_path, sitk.sitkLinear)
    # nii_path = ''
    # npy_path = ''
    # png_path = ''
    # slicing(nii_path,npy_path,png_path)
    # train_patient_folders = ['2503702','2835191','3911806','4230975','5049830']
    # copy_files_newfolder('/home/proxima-sx12/data/after_registration/CTA_MBF_npy', train_patient_folders,'/home/proxima-sx12/dataset/train')
    # test_patient_folders = ['1315171','4759812']
    # copy_files_newfolder('/home/proxima-sx12/data/after_registration/CTA_MBF_npy', test_patient_folders,'/home/proxima-sx12/dataset/val')
    # write_config_txt('/home/proxima-sx12/dataset/train','/home/proxima-sx12/dataset/val','/home/proxima-sx12/dataset/config')



