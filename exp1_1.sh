#!/bin/bash

for k in 32 64
do
	for l in 2 4 8 16
	do
		let poolEvery=${l}/2 
		./py-sbatch.sh -m hw2.experiments run-exp -n exp1_1_K${k}_L${l} --seed 42 --bs-train 128 --batches 100 --epochs 100 --early-stopping 5 --filters-per-layer ${k} --layers-per-block ${l} --pool-every ${poolEvery} --hidden-dims 100
	done 
done
