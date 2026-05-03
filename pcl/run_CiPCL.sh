#!bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --config-path configs\
    --config-name MSCOCO36_CiPCL.yaml \
    dataset_root="../DATA" \
    class_order="class_orders/MSCOCO36.yaml" \

