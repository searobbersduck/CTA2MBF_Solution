## 数据集

以`data_66_20210517`来讲解数据的目录结构：

```
tree -L 1
.
├── 0.ori_3                             
├── 3.sorted_dcm
├── 3.sorted_mask
├── 3.sorted_nii
├── 4.registration_batch
├── 5.mbf_myocardium
├── 6.inference
├── 6.inference_384x384x160
├── 6.inference_384x384x160_eval
├── 6.inference_384x384x160_train
├── 6.inference_slicemap
├── 7.analysis_result
├── annotation
└── 
```

### annotation
数据标注表格

### 0.ori_3
存储原始的数据，只根据series instance uid进行了文件夹的划分：
```
.
├── patient_id
│   ├── series_instance_uid0
│   ├── series_instance_uid0
|   ├── .....
├── patient_.....
```
```
.
├── 1791414
│   ├── 1.3.12.2.1107.5.1.4.75608.30000019021923054822600003249
│   ├── 1.3.12.2.1107.5.1.4.75608.30000019021923054822600003251
│   ├── 1.3.12.2.1107.5.1.4.75608.30000019021923054822600003300
│   ├── 1.3.12.2.1107.5.1.4.75608.30000019021923401028800062328
├── 1802920
│   ├── 1.3.12.2.1107.5.1.4.75608.30000019031923040590400003618
│   ├── 1.3.12.2.1107.5.1.4.75608.30000019031923040590400003622
│   ├── 1.3.12.2.1107.5.1.4.75608.30000019031923040590400003631
│   ├── 1.3.12.2.1107.5.1.4.75608.30000019031923234222200071115

```

---

### 3.sorted_dcm
从`0.ori_3`中提取出`CTA`/`BF`/`AVG`/`MIP`数据对, 存储为dicom格式
```
tree -L 2
.
├── 3057987
│   ├── AVG
│   ├── BF
│   ├── CTA
│   └── MIP
└── 3101737
    ├── AVG
    ├── BF
    ├── CTA
    └── MIP

```

---

### 3.sorted_mask

心脏腔室的分割结果，存储为`nii.gz`

```
tree -L 3
.
├── 3057987
    ├── AVG
    │   ├── AVG_MASK_connected.nii.gz       # 在分割结果的基础上，进行了最大连通域修正，这是之后用到的分割label
    │   ├── AVG_MASK.nii.gz                 # 分割结果
    │   └── AVG.nii.gz                      # 源数据
    └── CTA
        ├── CTA_MASK_connected.nii.gz
        ├── CTA_MASK.nii.gz
        └── CTA.nii.gz
└── 3101737
    ├── AVG
    │   ├── AVG_MASK_connected.nii.gz
    │   ├── AVG_MASK.nii.gz
    │   └── AVG.nii.gz
    └── CTA
        ├── CTA_MASK_connected.nii.gz
        ├── CTA_MASK.nii.gz
        └── CTA.nii.gz
```

腔室分割的效果如[图-腔室分割](https://github.com/searobbersduck/MedCommon/blob/main/experiments/seg/cardiac/chamber/readme.md)所示，其中的需要用到心肌部分的数据。

---

### 3.sorted_nii

从`0.ori_3`中提取出`CTA`/`BF`/`AVG`/`MIP`数据对, 存储为nii.gz格式

```
tree -L 2
.
├── 3057987
│   ├── AVG.nii.gz
│   ├── BF.nii.gz
│   ├── CTA.nii.gz
│   └── MIP.nii.gz
└── 3101737
    ├── AVG.nii.gz
    ├── BF.nii.gz
    ├── CTA.nii.gz
    └── MIP.nii.gz

```

---

### 4.registration_batch

配准结果，`cta_mip_avg.nii.gz`代表`avg图像配准到cta图像上`，配准矩阵是基于`mip图像配准到cta图像的配准矩阵`

```
tree -L 2
.
├── 3057987
│   ├── cta_mip_avg.nii.gz
│   ├── cta_mip_bf.nii.gz
│   ├── cta_mip_mip.nii.gz
│   └── cta.nii.gz
└── 3101737
    ├── cta_mip_avg.nii.gz
    ├── cta_mip_bf.nii.gz
    ├── cta_mip_mip.nii.gz
    └── cta.nii.gz

```

---

### 5.mbf_myocardium

根据`3.sorted_mask`中的mask，将心脏区域提取出来，用于之后的生成训练。`cropped_cta.nii.gz`和`cropped_mbf.nii.gz`即所用到的数据对。

```
tree -L 2
.
├── 3057987
│   ├── cropped_cta.nii.gz
│   ├── cropped_mask.nii.gz
│   ├── cropped_mbf.nii.gz
│   └── registration_cta_mip_bf_myocardium.nii.gz
└── 3101737
    ├── cropped_cta.nii.gz
    ├── cropped_mask.nii.gz
    ├── cropped_mbf.nii.gz
    └── registration_cta_mip_bf_myocardium.nii.gz

```

----

### 6.inference_384x384x160

按照384x384x160大小的数据块进行推断的结果，推断时，采用`model.eval()`的模式，训练时采用`model.train()`的模式

### 6.inference_384x384x160_train

按照384x384x160大小的数据块进行推断的结果，推断时，采用`model.train()`的模式，之所以会有如此的推断结果，因为训练时，采用`model.train()`的模式，而推断时采用`model.eval()`的模式，效果很差。

### 6.inference_384x384x160_eval

按照384x384x160大小的数据块进行推断的结果，推断时，采用`model.eval()`的模式, 训练时也使用`model.eval()`的模式

### 6.inference_slicemap

将推断结果，生成2D图片

---

### 7.analysis_result

记录推断数据的MAE结果，可以据此挑选效果比较好的数据



## 数据预处理

```
    root = '/data/medical/cardiac/cta2mbf/data_140_20210602'
    
    # # step 2 
    # # 将原始数据，根据series instance uid进行文件夹的划分
    # in_root = os.path.join(root, 'CTP数据')
    # out_root = os.path.join(root, '0.ori_3')
    # sort_by_series_uid_multiprocessing(in_root, out_root, 24)    

    # # step 3
    # # 从`0.ori_3`中提取出CTA/BF/AVG/MIP数据对, 存储为dicom格式
    # in_root = os.path.join(root, '0.ori_3')
    # out_root = os.path.join(root, '3.sorted_dcm')
    # preprocess_data_114_extract_modalitys(in_root, out_root)

    # # step 3
    # # CTA/BF/AVG/MIP数据对的dicom格式转换为nii.gz的方式进行存储
    # in_root = os.path.join(root, '3.sorted_dcm')
    # out_root = os.path.join(root, '3.sorted_nii')
    # step_3_4_convert_dicom_series_to_nii(in_root, out_root)
    
    # step 4 generate registration cmd 
    # # 将CTP-MIP图像配准到CTA影像上，并记录配准矩阵，并根据此配准矩阵实现BF到CTA的配准 
    '''
    配准过程执行data_preprocessing_registration.py
    '''

    # step 5 chamber segmentation
    # # 分割提取心脏腔室标签，目的是使用心肌的标签用来提取BF，利用其它结构的标签来框定生成训练所用到的数据的范围
    # data_root = os.path.join(root, '3.sorted_dcm')
    # out_dir = os.path.join(root, '3.sorted_mask')
    # cardiac_segmentation_new_algo(data_root, out_dir)
    # step_3_3_segment_cardiac_connected_region(root_dir = os.path.join(root, '3.sorted_mask'))

    # step 6 extract myocardium from bf images
    # # 根据配准结果以及腔室分割的心肌标签将心肌的区域提取出来
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
    # # 根据腔室分割的标签，提取出用于生成训练的心脏周边bbox数据
    mask_root = os.path.join(root, '3.sorted_mask')
    cta_root = mask_root
    mbf_root = os.path.join(root, '5.mbf_myocardium')
    out_root = os.path.join(root, '5.mbf_myocardium')
    mask_pattern = 'CTA/CTA_MASK_connected.nii.gz'
    # mbf_pattern = 'registration_bf_avg_myocardium.nii.gz'
    mbf_pattern = 'registration_cta_mip_bf_myocardium.nii.gz'
    step_5_2_extract_pericardium_bbox(mask_root, cta_root, mbf_root, out_root, mask_pattern, mbf_pattern)

```


## 文件夹结构形式

```
/data/medical/cardiac/cta2mbf$ tree -L 2
.
├── 20201216
│   ├── 0.config_info
│   ├── 0.ori
│   ├── 0.ori_2
│   ├── 0.ori_3
│   ├── 0.ori_step_2
│   ├── 3.sorted
│   ├── 3.sorted_mask
│   ├── 3.sorted_nii
│   ├── 3.sorted_npy
│   ├── 4.registration_batch1
│   └── 5.mbf_myocardium
├── 20210114
│   └── HXJ_FS_XJGNPG_SHLY_20210105
├── data_114_20210715
│   ├── 0.ori_3 -> /data/medical/cardiac/cta2mbf/data_114_20210318_old/0.ori_3
│   ├── 3.sorted_dcm
│   ├── 3.sorted_mask
│   ├── 3.sorted_nii
│   ├── 5.mbf_myocardium
│   └── annotation
├── data_140_20210602
│   ├── 0.ori_3
│   ├── 3.sorted_dcm
│   ├── 3.sorted_mask
│   ├── 3.sorted_nii
│   ├── 5.mbf_myocardium
│   └── data_wtf_127_20210702
├── data_66_20210517
│   ├── 0.ori_3
│   ├── 3.sorted_dcm
│   ├── 3.sorted_mask
│   ├── 3.sorted_nii
│   ├── 4.registration_batch
│   ├── 5.mbf_myocardium
│   ├── 6.inference
│   ├── 6.inference_384x384x160
│   ├── 6.inference_384x384x160_eval
│   ├── 6.inference_384x384x160_eval_slicemap
│   ├── 6.inference_384x384x160_train
│   ├── 6.inference_slicemap
│   ├── 7.analysis_result
│   ├── annotation
│   └── CTP灌注各类数据（李主任）
├── data_yourname
│   ├── 5.mbf_myocardium
│   └── checkpoints
└── ssl
    └── cropped_ori
```

```
tree -L 1
.
├── 20201216
├── 20210114
├── data_114_20210715
├── data_140_20210602
├── data_66_20210517
├── data_yourname
└── ssl

```

注意：所有文件夹中删除了`4.registration_batch`的子文件夹，该文件比较大，其目录结构如下：

```
/data/medical/cardiac/cta2mbf/data_66_20210517/4.registration_batch$ tree -L 2
.
├── 1023293
│   ├── cta_mip_avg.nii.gz
│   ├── cta_mip_bf.nii.gz
│   ├── cta_mip_mip.nii.gz
│   └── cta.nii.gz
├── 1037361
│   ├── cta_mip_avg.nii.gz
│   ├── cta_mip_bf.nii.gz
│   ├── cta_mip_mip.nii.gz
│   └── cta.nii.gz

```

### 文件夹具体对应表

1. <font color='red'>***20201216***</font>
```
# 对应文件： $MedDisk:\data\cardiac\cta2mbf\20201216
# cta2mbf_1,cta2mbf_2,cta2mbf_3,cta2mbf_4
/data/medical/cardiac/cta2mbf/20201216/0.ori$ ls
CTAMBF-11例  CTAMBF-14例  CTAMBF-26例  心肌灌注
```
2. <font color='red'>***20210114***</font>
```
# 对应文件：$MedDisk:\data\cardiac\cta2mbf\20210114
# cta2mbf_5
```
3. <font color='red'>***data_114_20210715/data_140_20210602***</font>
```
# 对应文件：$MedDisk:\data\cardiac\cta2mbf
# cta2mbf_6
```
4. <font color='red'>***data_66_20210517/6.inference_384x384x160_train***</font>
```
zip -r cta2mbf_inference_66.zip 6.inference_384x384x160_train 5.mbf_myocardium annotation 7.analysis_result
```