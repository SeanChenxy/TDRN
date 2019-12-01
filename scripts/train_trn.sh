#!/usr/bin/env bash
type='trn'
if [ $type = 'ssd' ]
then
    CUDA_VISIBLE_DEVICES='0,1,2,3' python ../train.py \
    --lr 0.002 \
    --warm_epoch 0 \
    --gamma 0.1 \
    --momentum 0.9 \
    --visdom 'no' \
    --save_folder '../weights/TRN/SSD' \
    --model_name ssd \
    --ssd_dim 320 \
    --step_list 130 170 190 \
    --save_interval 10 \
    --batch_size 32 \
    --augm_type 'ssd' \
    --num_workers 4 \
    --loss_coe 1.0 1.0 0.5 \
    --bn 'yes' \
    --gpu_ids '4,5,6,7' \
    --dataset_name 'VOC0712' \
    --set_file_name 'train_VID_DET' \
    --backbone 'VGG4s' \
    --c7_channel 1024 \
    --refine 'no' \
    --deform 0 \
    --multihead 'no' \
    --basenet 'vgg16bn_reducedfc.pth'
elif [ $type = 'drn' ]
then
    CUDA_VISIBLE_DEVICES='0,1,2,3' python ../train.py \
    --lr 0.002 \
    --warm_epoch 0 \
    --gamma 0.1 \
    --momentum 0.9 \
    --visdom 'no' \
    --save_folder '../weights/TRN/DRN' \
    --model_name 'drn' \
    --ssd_dim 320 \
    --step_list 130 170 210 \
    --save_interval 10 \
    --batch_size 32 \
    --augm_type 'ssd' \
    --num_workers 4 \
    --loss_coe 1.0 1.0 0.5 \
    --bn 'yes' \
    --dataset_name 'VOC0712' \
    --set_file_name 'train' \
    --backbone 'RefineDet_VGG' \
    --c7_channel 1024 \
    --refine 'yes' \
    --deform 4 \
    --multihead 'yes' \
    --basenet 'vgg16bn_reducedfc.pth'
elif [ $type = 'trn' ]
then
    CUDA_VISIBLE_DEVICES='0,1,2,3' python ../train_trn.py \
    --lr 0.002 \
    --gamma 0.1 \
    --momentum 0.9 \
    --visdom 'no' \
    --save_folder '../weights/TDRN_VGG' \
    --model_name 'trn' \
    --ssd_dim 320 \
    --step_list 50 70 90 \
    --save_interval 10 \
    --batch_size 32 \
    --augm_type 'ssd' \
    --num_workers 4 \
    --bn 'yes' \
    --gpu_ids '0,1,2,3' \
    --dataset_name 'VIDDET' \
    --set_file_name 'train_VID_DET' \
    --backbone 'VGG' \
    --c7_channel 1024 \
    --deform 'yes' \
    --loose 1. \
    --basenet 'vgg16bn_reducedfc.pth' \
    --resume_static '../weights/VID/ssd320_VGG16_6298.pth'
elif [ $type = 'trn_mob' ]
then
    CUDA_VISIBLE_DEVICES='0,1,2,3' python ../train_trn.py \
    --lr 0.002 \
    --gamma 0.1 \
    --momentum 0.9 \
    --visdom 'no' \
    --save_folder '../weights/TRN_MobileMet' \
    --model_name 'trn' \
    --ssd_dim 320 \
    --step_list 60 80 100 \
    --save_interval 10 \
    --batch_size 32 \
    --augm_type 'ssd' \
    --num_workers 4 \
    --bn 'yes' \
    --dataset_name 'VIDDET' \
    --set_file_name 'train_VID_DET' \
    --backbone 'MobNet' \
    --c7_channel 1024 \
    --deform 'no' \
    --loose 1. \
    --basenet 'mobilenet_reducedfc.pth' \
    --resume_static '../weights/VID/ssd320_MobileNet_5829.pth'
fi
