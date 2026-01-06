#!/bin/bash

if [ "$#" -eq 1 ]; then
    PARAM_FILE="$1"
	sbatch syn_census/synthetic_data_generation/hh_microdata.sh "$PARAM_FILE" simple_1;
	sbatch syn_census/synthetic_data_generation/hh_microdata.sh "$PARAM_FILE"  simple_2;
else
    echo Missing arguments PARAM_FILE
    exit 1
fi

