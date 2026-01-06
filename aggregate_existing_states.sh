#!/bin/bash

for datafile in *param.json
do 
	sbatch shard_aggregation.sh $datafile simple_1;
	sbatch shard_aggregation.sh $datafile simple_2;
done
