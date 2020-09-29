
## 3D心肌模型相关
----
### predict
分割心肌，执行如下代码：

```
CUDA_VISIBLE_DEVICES=4 python train_unet3d.py inference '../data/seg/nii_file/1.3.12.2.1107.5.1.4.60320.30000015012900333934300003426/img.nii.gz' '../data/seg/model/cardiac_seg_train_0.013_val_0.020' '../data/seg/inference/test'
```

其中， '../data/seg/inference/test'为输出结果路径，路径中`mask_pred.nii.gz`即为分割结果

----

