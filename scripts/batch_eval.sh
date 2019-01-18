#!/usr/bin/env bash
conf_thresh=0.01
nms_thresh=0.45
top_k=200
type='ssd_refine_coco'
for iter in {75,}
do
    if [ $type = 'ssd_refine' ]
    then
        CUDA_VISIBLE_DEVICES='6' python ../evaluate.py \
        --model_dir '../weights' \
        --model_name ssd \
        --ssd_dim 320 \
        --iteration $iter \
        --save_folder '../eval/VOC07' \
        --dataset_name 'VOC0712' \
        --year '2007' \
        --confidence_threshold $conf_thresh \
        --nms_threshold $nms_thresh \
        --top_k $top_k \
        --backbone 'RefineDet_VGG' \
        --bn 'yes' \
        --refine 'yes' \
        --deform 1 \
        --multihead 'yes' \
        --c7_channel 1024 \
        --set_file_name 'test' \
        --gpu_id '4' \
        --detection 'yes' \
        --cuda 'yes' \
        --display 'no'
    elif [ $type = 'ssd_refine_voc12' ]
    then
        CUDA_VISIBLE_DEVICES='3' python ../evaluate.py \
        --model_dir '../weights' \
        --model_name ssd \
        --ssd_dim 320 \
        --iteration 210 \
        --save_folder '../eval/VOC12' \
        --dataset_name 'VOC0712' \
        --year '2012' \
        --confidence_threshold $conf_thresh \
        --nms_threshold $nms_thresh \
        --top_k $top_k \
        --backbone 'RefineDet_VGG' \
        --bn 'yes' \
        --refine 'yes' \
        --deform 1 \
        --multihead 'yes' \
        --c7_channel 1024 \
        --set_file_name 'test' \
        --tssd 'ssd' \
        --detection 'yes' \
        --cuda 'yes'
    elif [ $type = 'ssd_refine_coco' ]
    then
        CUDA_VISIBLE_DEVICES='3'  python ../evaluate_coco.py \
        --model_dir '../weights' \
        --model_name ssd \
        --ssd_dim 320 \
        --iteration 150 \
        --save_folder '../eval/COCO' \
        --dataset_name 'COCO' \
        --year 'trainval2014' \
        --confidence_threshold $conf_thresh \
        --nms_threshold $nms_thresh \
        --top_k 300 \
        --backbone 'RefineDet_VGG' \
        --bn 'yes' \
        --refine 'yes' \
        --deform 1 \
        --multihead 'yes' \
        --c7_channel 1024 \
        --set_file_name 'minival2014' \
        --detection 'yes' \
        --display 'no' \
        --cuda 'yes'
    fi
done
