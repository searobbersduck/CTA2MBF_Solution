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
└── CTP灌注各类数据（李主任）
```

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