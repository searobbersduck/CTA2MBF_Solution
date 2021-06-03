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
2. 数据说明：[readme.md](data_preprocessing/readme.md)