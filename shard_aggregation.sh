#!/bin/bash
#SBATCH -c 1                # Number of cores (-c)
#SBATCH -t 0-0:30          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p sched_mit_sloan_batch_r8 # Partition to submit to
#SBATCH --mem=10G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o out_files/samp.%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e out_files/samp.%j.err  # File to which STDERR will be written, %j inserts jobid
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
source activate synthetic-census
python3 aggregate_data_shards.py --from_params "$PARAM_FILE" --task_name "$TASK_NAME"
