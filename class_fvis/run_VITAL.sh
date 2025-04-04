#!/bin/bash

arch_name=resnet50 # architecture name
rand=50  # number of random images
cls=1  # target class neuron to visualize 
run_id=1  # run number

python class_neurons_fvis.py \
    --arch_name $arch_name --target $cls --run_id $run_id --num_real_img $rand --gpuid 0

