#!/bin/bash

CUDA_VISIBLE_DEVICES=1,2,3,4 python main.py \
    --batch_size 1200 \
    # --num_layers_AE 1