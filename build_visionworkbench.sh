#!/bin/bash

isMac=$(uname -s | grep Darwin)
if [ "$isMac" != "" ]; then
  cc_comp=clang
  cxx_comp=clang++
else
  cc_comp=x86_64-conda_cos6-linux-gnu-gcc
  cxx_comp=x86_64-conda_cos6-linux-gnu-g++
fi

buildDir=$PWD
envPath=/explore/nobackup/people/cssprad1/micromamba/envs/aspv2
# git clone                            \
#     git@github.com:visionworkbench/visionworkbench.git
cd visionworkbenchGPU
# Uncomment below if desired to build a specific version
# git checkout 3.3.0
mkdir -p build
cd build
$envPath/bin/cmake ..                             \
  -DASP_DEPS_DIR=$envPath                         \
  -DCMAKE_VERBOSE_MAKEFILE=ON                     \
  -DCMAKE_INSTALL_PREFIX=$buildDir/install        \
  -DCMAKE_C_COMPILER=${envPath}/bin/$cc_comp      \
  -DCMAKE_CXX_COMPILER=${envPath}/bin/$cxx_comp   \
  -DCMAKE_CUDA_COMPILER=/panfs/ccds02/app/modules/nvidia/platform/x86_64/rhel/8.8/12.1/Linux_x86_64/23.5/compilers/bin/nvcc
make -j10 && make install
