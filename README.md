# CTA2MBF_Solution
a solution for generating BF image from CTA image

----
> 道德三皇五帝，功名夏后商周。五霸七雄闹春秋，秦汉兴亡过手。
> 
> 青史几行名姓，北邙无数荒丘。前人耕种后人收，说甚龙争虎斗。
----

## 数据

1. 数据预处理[data_preprocessing.py](./data_preprocessing/data_preprocessing.py)
    1. 数据整理
    2. 将数据按pid/series_uids进行划分
    3. 提取其中的CTA/MBF/MIP/AVG数据对
    4. 配准
    5. 分割心脏腔室
    6. 提取心肌部分的MBF影像
    7. 提取包含心肌bbox的CTA/MBF/MIP/AVG数据对
        
        * 配准效果图
        ![4.配准](data_preprocessing/img/1023293/register_bf2cta.gif)

        * 根据配准结果以及腔室分割的心肌标签将心肌的区域提取出来
        ![4.配准](data_preprocessing/img/1023293/register_bf2cta_myocardium.gif)

        * 根据腔室分割的标签，提取出用于生成训练的心脏周边bbox数据
        ![4.配准](data_preprocessing/img/1023293/register_bf2cta_myocardium_cropped.gif)

2. 数据说明：[readme.md](data_preprocessing/readme.md)

## 模型训练

### BF自监督模型训练

### L1 Loss

脚本参见:[train_mbf.sh](train/train_mbf.sh)

### L1 Loss + Mask Loss

脚本参见:[train_mbf.sh](train/train_mbf.sh)

### L1 Loss + Mask Loss + Matching Loss (self supervised)

todo:

## 模型推断

1. 脚本参见:[inference.sh](train/inference.sh)
2. 将3D数据转换成2D数据进行展示：
3. 计算生成数据的指标：todo


