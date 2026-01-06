#!/bin/bash
#SBATCH -c 4                # Number of cores (-c)
#SBATCH -t 0-00:30          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p sched_mit_sloan_batch_r8   # Partition to submit to
#SBATCH --mem=10G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --mail-type=END
if [ "$#" -eq 2 ]; then
    PARAM_FILE="$1"
    TASK_NAME="$2"
else
    echo Missing arguments PARAM_FILE or TASK_NAME
    exit 1
fi
module load miniforge
conda activate synthetic-census
python3 syn_census/synthetic_data_generation/hh_to_person_microdata.py --from_params "$PARAM_FILE" --task_name "$TASK_NAME"
