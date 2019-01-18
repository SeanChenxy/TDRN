#!/usr/bin/env bash
conf_thresh=0.01
nms_thresh=0.45
top_k=200
type='trn_batch'
if [ $type = 'fpn' ]
then
    CUDA_VISIBLE_DEVICES='5' python ../evaluate.py \
    --model_dir '../weights040/TRN/FPN320VggBn_VIDDET3' \
    --model_name ssd \
    --ssd_dim 320 \
    --iteration 100 \
    --save_folder '../eval/VID' \
    --dataset_name 'VID2017' \
    --year '2007' \
    --confidence_threshold $conf_thresh \
    --nms_threshold $nms_thresh \
    --top_k $top_k \
    --backbone 'RefineDet_VGG' \
    --bn 'yes' \
    --refine 'no' \
    --deform 0 \
    --multihead 'no' \
    --c7_channel 1024 \
    --attention 'no' \
    --res_attention 'no' \
    --channel_attention 'no' \
    --pm 0.0 \
    --set_file_name 'val' \
    --tssd 'ssd' \
    --gpu_id '6' \
    --detection 'yes' \
    --cuda 'yes'
elif [ $type = 'ssd' ]
then
    CUDA_VISIBLE_DEVICES='4' python ../evaluate.py \
    --model_dir '../weights040/TRN/SSD320VggBn_VIDDET' \
    --model_name ssd \
    --ssd_dim 320 \
    --iteration 100 \
    --save_folder '../eval/VID' \
    --dataset_name 'VID2017' \
    --year '2017' \
    --confidence_threshold $conf_thresh \
    --nms_threshold $nms_thresh \
    --top_k $top_k \
    --backbone 'VGG4s' \
    --bn 'yes' \
    --refine 'no' \
    --deform 0 \
    --multihead 'no' \
    --c7_channel 1024 \
    --attention 'no' \
    --res_attention 'no' \
    --channel_attention 'no' \
    --pm 0.0 \
    --set_file_name 'val' \
    --tssd 'ssd' \
    --detection 'yes' \
    --cuda 'yes' \
    --display 'no'
elif [ $type = 'drn' ]
then
    CUDA_VISIBLE_DEVICES='7' python ../evaluate.py \
    --model_dir '../weights040/TRN/DRN512VggBnMultiDef_VOC' \
    --model_name drn \
    --ssd_dim 512 \
    --iteration 180 \
    --save_folder '../eval/VID' \
    --dataset_name 'VOC0712' \
    --year '2007' \
    --confidence_threshold $conf_thresh \
    --nms_threshold $nms_thresh \
    --top_k $top_k \
    --backbone 'RefineDet_VGG' \
    --bn 'yes' \
    --refine 'yes' \
    --deform 4 \
    --multihead 'yes' \
    --c7_channel 1024 \
    --attention 'no' \
    --res_attention 'no' \
    --channel_attention 'no' \
    --pm 0.0 \
    --set_file_name 'test' \
    --tssd 'ssd' \
    --detection 'yes' \
    --cuda 'yes' \
    --display 'no'
elif [ $type = 'trn' ]
then
    CUDA_VISIBLE_DEVICES='6' python ../evaluate_trn.py \
    --static_dir '../weights040/TRN/SSD320Mob_VIDDET/ssd320_VIDDET_100.pth' \
    --trn_dir '../weights040/TRN/TRN320Mob_VIDDET2/trn320_VIDDET_80.pth' \
    --model_name ssd \
    --ssd_dim 320 \
    --save_folder '../eval/VID' \
    --dataset_name 'VID2017' \
    --year '2007' \
    --confidence_threshold $conf_thresh \
    --nms_threshold $nms_thresh \
    --top_k $top_k \
    --backbone 'MobNet' \
    --bn 'yes' \
    --deform 'yes' \
    --c7_channel 1024 \
    --set_file_name 'val' \
    --detection 'yes' \
    --interval 3 \
    --loose 1. \
    --cuda 'yes' \
    --display 'no'
elif [ $type = 'trn_batch' ]
then
    for int in {8,}
    do
        CUDA_VISIBLE_DEVICES='7' python ../evaluate_trn.py \
        --static_dir '../weights040/TRN/SSD320Mob_VIDDET/ssd320_VIDDET_100.pth' \
        --trn_dir '../weights040/TRN/TRN320Mob_VIDDET3/trn320_VIDDET_100.pth' \
        --model_name ssd \
        --ssd_dim 320 \
        --save_folder '../eval/VID' \
        --dataset_name 'VID2017' \
        --year '2017' \
        --confidence_threshold $conf_thresh \
        --nms_threshold $nms_thresh \
        --top_k $top_k \
        --backbone 'MobNet' \
        --bn 'yes' \
        --deform 'no' \
        --loose 0.5 \
        --c7_channel 1024 \
        --set_file_name 'val' \
        --detection 'yes' \
        --interval $int \
        --cuda 'yes' \
        --display 'no'
    done
fi
#     --trn_dir '../weights040/TRN/TRN320VggBn_VID2017/trn320_seqVID2017_70.pth' \
#     --trn_dir '../weights040/TRN/TRN320VggBn_VIDDET3/trn320_VIDDET_50.pth' \
#     --static_dir '../weights040/TRN/SSD320VggBn_VIDDET/ssd320_VIDDET_100.pth' \
