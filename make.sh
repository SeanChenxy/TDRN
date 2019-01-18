#!/usr/bin/env bash
cd ./utils/

CUDA_PATH=/usr/local/cuda/

python build.py build_ext --inplace

cd ..

cd ./utils/deformconv/
nvcc -c -o deform_conv_cuda_kernel.cu.o deform_conv_cuda_kernel.cu -x cu -Xcompiler -fPIC -std=c++11
cd ..
CC=g++ python build_deformconv.py
cd ..
