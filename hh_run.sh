#!/bin/bash

if [ "$#" -eq 2 ]; then
    PARAM_FILE="$1"
    TASK_NAME="$2"
	sbatch hh_microdata.sh "$PARAM_FILE" "$TASK_NAME" simple_1;
	sbatch hh_microdata.sh "$PARAM_FILE" "$TASK_NAME" simple_2;
else
    echo Missing arguments PARAM_FILE or TASK_NAME
    exit 1
fi

done
