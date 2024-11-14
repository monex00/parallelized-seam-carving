#!/bin/bash
#SBATCH --job-name=compile_and_run_matrix_stencil
#SBATCH --output=output1.txt
#SBATCH --error=error1.txt
#SBATCH --partition=cascadelake
#SBATCH --gres=gpu:1             # Richiede 1 GPU


# Compilazione
srun g++ row_seq.cpp -o row_seq 
if [ $? -ne 0 ]; then
    echo "Compilazione fallita"
    exit 1
fi

# Esecuzione
srun ./row_seq
