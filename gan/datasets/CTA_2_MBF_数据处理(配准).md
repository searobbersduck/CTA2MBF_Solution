[TOC]
#### 配准前处理
##### 预处理函数
定义在data_preprocessing.py 
位置在/home/zhangwd/code/work/CTA2MBF_Solution/gan/datasets/preprocessing/
###### 抓取CTA/MIP/MBF序列的函数
```python
extract_from_hospital_folder(in_folder, out_folder, series_category)   
```
###### 检查MBF和CTA采集时间的函数
```python
check_acq_time(cta_path,mbf_path)
```
###### dicom转nii.gz的函数
```python
dcm_to_nii(in_dcm_path, out_nii_path,series_category)
```
###### 重采样函数
```python
resample_sitkImage_by_spacing(sitkImage, newSpacing, vol_default_value='min', interpolator=sitk.sitkLinear)
#输入文件夹包含cta/mip/mbf的nii格式
resample(in_folder,out_folder, interpolator_method) 
```
##### 配准预处理
###### 1. 读医院数据文件夹里的dicom，取病人编号一致且series名称是CTA/MIP/MBF的复制到新的文件夹
```python
in_folder = 'D:/sixth_hospital_20200616_CTP/DICOM/six_ctp'
MIP_folder ='D:/MIP'
MBF_folder = 'D:/MBF'
CTA_folder = 'D:/CTA'
extract_from_hospital_folder(in_folder, MIP_folder, 'MIP')
extract_from_hospital_folder(in_folder, MBF_folder, 'MBF')
extract_from_hospital_folder(in_folder, CTA_folder, 'CTA')
```
###### 2. 输出CTA和MBF的采集时间，看是否相近
````python
check_acq_time(CTA_folder,MBF_folder)
```
###### 3. 读CTA/MIP/MBF文件夹里的dicom封装成一个对象，转换为nii.gz去配准和计算
```python
nii_path =  'D:/nii'
dcm_to_nii(CTA_folder, nii_path,'CTA')
dcm_to_nii(MIP_folder, nii_path,'MIP')
dcm_to_nii(MBF_folder, nii_path,'MBF')
```
###### 4. 将CTA重采样到 isotropy，然后以CTA的spacing为标准，重采样MIP/MBF
```python
resample_nii_path = 'D:/resample_nii'
resample(nii_path, resample_nii_path, sitk.sitkLinear)
```
#### 配准
##### 配准过程有关函数
定义在regis.py文件
位置在/home/zhangwd/code/work/CTA2MBF_Solution/gan/datasets/registration/
###### 3D图显示：两张并排显示以及叠加显示（jupyter）
```python
# Callback invoked by the interact IPython method for scrolling through the image stacks of the two images (moving and fixed).
def display_images(fixed_image_z, moving_image_z, fixed_npa, moving_npa):
    # Create a figure with two subplots and the specified size.
    plt.subplots(1,2,figsize=(10,8))
    
    # Draw the fixed image in the first subplot.
    plt.subplot(1,2,1)
    plt.imshow(fixed_npa[fixed_image_z,:,:],cmap=plt.cm.Greys_r);
    plt.title('fixed image')
    plt.axis('off')
    
    # Draw the moving image in the second subplot.
    plt.subplot(1,2,2)
    plt.imshow(moving_npa[moving_image_z,:,:],cmap=plt.cm.Greys_r);
    plt.title('moving image')
    plt.axis('off')
    
    plt.show()


# Callback invoked by the IPython interact method for scrolling and modifying the alpha blending of an image stack of two images that occupy the same physical space. 
def display_images_with_alpha(image_z, alpha, fixed, moving):
    img = (1.0 - alpha)*fixed[:,:,image_z] + alpha*moving[:,:,image_z] 
    plt.imshow(sitk.GetArrayViewFromImage(img),cmap=plt.cm.Greys_r);
    plt.axis('off')
    plt.show()
```
###### 初配准，返回变换矩阵
```python
initial_registration(fixed_image, moving_image, initial_transform)
```
###### 执行变换并返回配准图像
```
perform_transform(transform, fixed_image, moving_image)
```
###### ffd模型
```python
bspline_registration(fixed_image, moving_image, fixed_image_mask=None, fixed_points=None, moving_points=None)
# ffd model with Multi-resolution control point grid
bspline_registration_morepoint(fixed_image, moving_image, fixed_image_mask=None, fixed_points=None, moving_points=None)
```
###### 三步配准过程封装成函数
1）直接ffd配准
2）ffd with a multi-resolution control point grid模型配准
3）1）的配准结果作为初始值，执行2）
```python
registration_three_phase(nii_path, save_folder)
```
##### 配准过程1：Affine+FFD
###### 加载CTA作为fixed图像，MIP/MBF为待配准图像
`````python
cta_path = '/home/proxima-sx12/dataset/cta_mip_rasample/4759812/resampled_CTA_512_512_276.nii.gz'
mip_path = '/home/proxima-sx12/dataset/cta_mip_rasample/4759812/resampled_MIP_451_451_234.nii.gz'
mbf_path = '/home/proxima-sx12/dataset/cta_mip_rasample/4759812/resampled_MBF_451_451_234.nii.gz'


fixed_image =  sitk.ReadImage(cta_path, sitk.sitkFloat32)
moving_image = sitk.ReadImage(mip_path, sitk.sitkFloat32) 
mbf_image = sitk.ReadImage(mbf_path, sitk.sitkFloat32) 
```
###### 定义仿射变换，执行仿射变换的配准
```python
affineTx = sitk.CenteredTransformInitializer(fixed_image, moving_image,
                                                            sitk.AffineTransform(
                                                                fixed_image.GetDimension()))
affine_transform = initial_registration(fixed_image, moving_image, affineTx)
mip_affine= perform_transform(affine_transform, fixed_image, moving_image)
mbf_affine = perform_transform(affine_transform, fixed_image, mbf_image)
```
###### 将仿射变换配准后的图作为moving image，执行FFD模型的配准
```python
ffd_transform,final_metric = bspline_registration(fixed_image = fixed_image, 
                                      moving_image = mip_affine,
                                      fixed_image_mask = None,
                                      fixed_points = None, 
                                      moving_points = None
                                     )
mip_ffd = perform_transform(ffd_transform,fixed_image,mip_affine)
mbf_ffd = perform_transform(ffd_transform, fixed_image,mbf_affine)
```
###### 展示配准效果，保存变换矩阵、配准后的MIP和MBF
```python
#成对展示CTA_MBF，观察配准效果 (jupyter)
interact(display_images, fixed_image_z=(0,fixed_image.GetSize()[2]-1), moving_image_z=(0,mbf_ffd.GetSize()[2]-1), fixed_npa = fixed(sitk.GetArrayViewFromImage(fixed_image)), moving_npa=fixed(sitk.GetArrayViewFromImage(mbf_ffd)));
interact(display_images_with_alpha, image_z=(0,fixed_image.GetSize()[2]-1), alpha=(0.0,1.0,0.05), fixed = fixed(fixed_image), moving=fixed(mbf_ffd));
#保存结果
sitk.WriteTransform(ffd_transform, 'name.tfm')
sitk.WriteImage(mip_ffd, 'name.nii.gz')
sitk.WriteImage(mbf_ffd, 'name.gz')
```
##### 配准过程2：FFD
###### 加载cta/mip/mbf的nii文件夹路径和保存路径
```python
nii_path = '/home/proxima-sx12/dataset/cta_mip_rasample'
save_folder = '/home/proxima-sx12/regis_results'
```
###### 批量处理：每个病人执行三次配准
```python
registration_three_phase(nii_path, save_folder)
```
#### 配准后将nii切片存为numpy和png格式
```python
#nii_path为cta和mbf nii格式的地址
slicing(nii_path,npy_path,png_path)
```
代码位置在/home/zhangwd/code/work/CTA2MBF_Solution/gan/datasets/preprocessing/data_preprocessing.py 
#### 将numpy格式的数据制作成 train/val/config
```python
#训练集病人id
train_patient_folders = ['2503702','2835191','3911806','4230975','5049830']
copy_files_newfolder('/home/proxima-sx12/data/after_registration/CTA_MBF_npy', train_patient_folders,'/home/proxima-sx12/dataset/train')
#测试集病人id
test_patient_folders = ['1315171','4759812']
test_patient_folders,'/home/proxima-sx12/dataset/val')
write_config_txt('/home/proxima-sx12/dataset/train','/home/proxima-sx12/dataset/val','/home/proxima-sx12/dataset/config')
```
代码位置在/home/zhangwd/code/work/CTA2MBF_Solution/gan/datasets/preprocessing/data_preprocessing.py 


