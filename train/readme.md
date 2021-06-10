## L1 Loss

```
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch \
--master_addr='10.100.37.21' \
--master_port='29503' \
--nproc_per_node=2 \
--nnodes=1 \
--use_env \
train.py \
--dataroot /data/medical/cardiac/cta2mbf/data_114_20210318/5.mbf_myocardium \
--model pix2pix_3d \
--input_nc 1 \
--output_nc 1 \
--ngf 32 \
--netG resnet_6blocks \
--ndf 8 \
--no_dropout \
--netD pixel \
--norm batch \
--display_server='10.100.37.21' \
--display_port=8098 \
--display_id=1 \
--lambda_L1=1 \
--n_epochs=500 \
--display_freq=10 \
--print_freq=10 \
--save_epoch_freq=10 \
--lr_policy cosine \
--lr 1e-4 \
--checkpoints_dir /data/medical/cardiac/cta2mbf/data_114_20210318/checkpoints \
--name cta2mbf \
--crop_size 352 352 160 \
--dst_vis_lut jet \
--src_pattern cropped_cta.nii.gz \
--dst_pattern cropped_mbf.nii.gz \
--continue_train
```

## L1 Loss + Mask

```
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch \
--master_addr='10.100.37.21' \
--master_port='29503' \
--nproc_per_node=2 \
--nnodes=1 \
--use_env \
train.py \
--dataroot /data/medical/cardiac/cta2mbf/data_114_20210318/5.mbf_myocardium \
--model pix2pix_3d \
--input_nc 1 \
--output_nc 1 \
--ngf 32 \
--netG resnet_6blocks \
--ndf 8 \
--no_dropout \
--netD pixel \
--norm batch \
--display_server='10.100.37.21' \
--display_port=8098 \
--display_id=1 \
--lambda_L1=1 \
--n_epochs=500 \
--display_freq=10 \
--print_freq=10 \
--save_epoch_freq=10 \
--lr_policy cosine \
--lr 1e-4 \
--checkpoints_dir /data/medical/cardiac/cta2mbf/data_114_20210318/checkpoints \
--name cta2mbf \
--crop_size 64 64 64 \
--dst_vis_lut jet \
--src_pattern cropped_cta.nii.gz \
--dst_pattern cropped_mbf.nii.gz \
--mask_pattern cropped_mask.nii.gz \
--mask_label 6 \
--lambda_L1_Mask 0.4 \
--continue_train
```