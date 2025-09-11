#!/bin/bash
#SBATCH -c 4                # Number of cores (-c)
#SBATCH -t 0-16:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p sched_mit_sloan_batch_r8   # Partition to submit to
#SBATCH --mem=5000           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o out_files/census.%A_%a.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e out_files/census.%A_%a.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --array=1-400
#SBATCH --mail-type=END
if [ "$#" -eq 2 ]; then
    PARAM_FILE="$1"
    TASK_NAME="$2"
else
    echo Missing arguments PARAM_FILE or TASK_NAME
    exit 1
fi
module load miniforge
module load gurobi/12.0.3
conda activate synthetic-census
python3 generate_data_shard.py --from_params "$PARAM_FILE" --task $SLURM_ARRAY_TASK_ID --num_tasks $SLURM_ARRAY_TASK_COUNT --task_name "$TASK_NAME"
