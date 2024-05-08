#!/bin/bash

install_base="/home/mrowkam/installs/"

# Build the executables for each backend type
for backend in  CUDA
do
    
    if [ -d "build-"$backend ]; then 
        rm -rf "build-"$backend 
    fi 

    mkdir "build-"$backend 
    cd "build-"$backend 

    if [ "$backend" == "CUDA" ]; then
        cmake -DKokkos_ROOT=$install_base"KOKKOS_"$backend -DMParT_ROOT=$install_base"MPART_"$backend -DCMAKE_CXX_COMPILER="$install_base"KOKKOS_CUDA/bin/nvcc_wrapper ../
        make
    else
        cmake -DKokkos_ROOT=$install_base"KOKKOS_"$backend -DMParT_ROOT=$install_base"MPART_"$backend ../
        make
    fi 

    cd ../

done


# Run the serial tests 
cd build-SERIAL 
#./SampleSpeed SERIAL 
cd ../ 

export OMP_PROC_BIND=spread
export OMP_PLACES=threads

# Run the host multithreaded backends
for backend in OPENMP
do 

    cd "build-"$backend 

    for num_threads in 32
    do 
        ./SampleSpeed $backend $num_threads
    done 

    cd ../ 
done 

# Run the CUDA backend 
cd build-CUDA 
./SampleSpeed CUDA 
cd ../

# Collect all the results in the "results" folder
 
if [ -d results ]; then 
    rm -rf results 
fi 
mkdir results

for backend in SERIAL THREADS OPENMP CUDA
do
    cp "build-"$backend/ST_CPP_*.txt results/
done 

lscpu > results/cpu_spec.txt 
nvcc --version > results/nvcc_spec.txt
nvidia-smi > results/gpu_spec.txt




