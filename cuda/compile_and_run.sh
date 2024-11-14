#!/bin/bash
#SBATCH --job-name=compile_and_run_matrix_stencil
#SBATCH --output=output.txt
#SBATCH --error=error.txt
#SBATCH --partition=cascadelake
#SBATCH --gres=gpu:1             # Richiede 1 GPU


# Compilazione
srun nvcc row2.cu -o row2 -lcudart
if [ $? -ne 0 ]; then
    echo "Compilazione fallita"
    exit 1
fi

# Esecuzione
srun ./row2
