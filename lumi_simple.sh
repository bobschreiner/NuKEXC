#!/bin/bash -l
#SBATCH --job-name=NUKEXC_TEST # Job name
#SBATCH --output=NUKEXC_TEST.o%j # Name of stdout output file
#SBATCH --error=NUKEXC_TEST.e%j  # Name of stderr error file
#SBATCH --partition=standard-g  # partition name
#SBATCH --nodes=1               # Total number of nodes 
#SBATCH --ntasks-per-node=1     # 8 MPI ranks per node, 16 total (2x8)
#SBATCH --gpus-per-node=1       # Allocate one gpu per MPI rank
#SBATCH --time=00:10:00       # Run time (d-hh:mm:ss)
#SBATCH --account=project_462001268  # Project for billing

srun ctest
