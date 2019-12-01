#!/usr/bin/env bash
type='ssd_refine'
if [ $type = 'ssd_refine' ]
then
    CUDA_VISIBLE_DEVICES='0,1,2,3' python ../train.py \
    --lr 0.001 \
    --warm_epoch 0 \
    --gamma 0.1 \
    --momentum 0.9 \
    --visdom 'no' \
    --save_folder '../weights/VOC' \
    --model_name ssd \
    --ssd_dim 320 \
    --step_list 130 160 190 \
    --save_interval 10 \
    --batch_size 32 \
    --augm_type 'ssd' \
    --num_workers 2 \
    --loss_coe 1.0 1.0 0.5 \
    --bn 'yes' \
    --gpu_ids '0,1,2,3' \
    --dataset_name 'VOC0712' \
    --set_file_name 'train' \
    --backbone 'RefineDet_VGG' \
    --c7_channel 1024 \
    --refine 'yes' \
    --deform 1 \
    --multihead 'yes' \
    --basenet 'vgg16bn_reducedfc.pth'
elif [ $type = 'ssd_refine_coco' ]
then
    CUDA_VISIBLE_DEVICES='0,1,2,3' python ../train.py \
    --lr 0.001 \
    --warm_epoch 0 \
    --gamma 0.1 \
    --momentum 0.9 \
    --visdom 'no' \
    --save_folder '../weights/COCO' \
    --model_name ssd \
    --ssd_dim 320 \
    --step_list 90 120 150 \
    --save_interval 5 \
    --batch_size 32 \
    --augm_type 'ssd' \
    --num_workers 4 \
    --loss_coe 1.0 1.0 0.5 \
    --bn 'yes' \
    --dataset_name 'COCO' \
    --set_file_name 'train' \
    --backbone 'RefineDet_VGG' \
    --c7_channel 1024 \
    --refine 'yes' \
    --deform 1 \
    --multihead 'yes' \
    --basenet 'vgg16bn_reducedfc.pth'
elif [ $type = 'ssd_refine_voc12' ]
then
    # data/config.py:
    # 'VOC0712':([('2007', 'trainval'), ('2012', 'trainval'), ('2007', 'test')], len(VOC_CLASSES) + 1, VOCroot)
    CUDA_VISIBLE_DEVICES='0,1,2,3' python ../train.py \
    --lr 0.002 \
    --warm_epoch 0 \
    --gamma 0.1 \
    --momentum 0.9 \
    --visdom 'no' \
    --save_folder '../weights/VOC12' \
    --model_name ssd \
    --ssd_dim 320 \
    --step_list 150 190 210 \
    --save_interval 10 \
    --batch_size 32 \
    --augm_type 'ssd' \
    --num_workers 2 \
    --loss_coe 1.0 1.0 0.5 \
    --bn 'yes' \
    --gpu_ids '0,1,2,3' \
    --dataset_name 'VOC0712' \
    --set_file_name 'train' \
    --backbone 'RefineDet_VGG' \
    --c7_channel 1024 \
    --refine 'yes' \
    --deform 1 \
    --multihead 'no' \
    --basenet 'vgg16bn_reducedfc.pth'
fi
