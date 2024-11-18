#!/bin/bash
#SBATCH --job-name=SCS_mpi               # Job name for world_size=2
#SBATCH --nodes=4                      # Request 1 node
#SBATCH --ntasks=4                      # Total 1 MPI task
#SBATCH --cpus-per-task=1                # 1 CPU per task
#SBATCH --gres=gpu:1                 # 1 GPU per node
#SBATCH --time=00:20:00                  # Time limit
#SBATCH --partition=amperenodes          # GPU partition

# Load necessary modules
module load OpenMPI/4.1.5-GCC-12.3.0
module load CUDA

# Set OpenMPI environment variables
export OMPI_MCA_btl_base_verbose=30
export OMPI_MCA_plm_base_verbose=5
export OMPI_MCA_btl=^openib
export UCX_NET_DEVICES=mlx5_0:1
export UCX_TLS=rc,tcp,sm
export OMPI_MCA_pml=ucx

# Print debug information
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "Number of nodes: $SLURM_NNODES"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"


# Change to the directory containing the SCS executable
cd /data/user/kefan/Wajid/finalSCS/dist/




# Define the datasets to run
datasets=( "amazon_q" )

# Loop through the datasets and run the program
for dataset in "${datasets[@]}"; do
    echo "Running for dataset: $dataset"
    # Run the program using srun with absolute paths
    srun --mpi=pmix \
         --distribution=block \
         --cpu-bind=cores \
         ./SCS \
         ../../data/edgeList/$dataset \
         140000 100000 0.1 1 10 100 1 1 1 1 1 \
         ./client/query/exp9/$dataset
done