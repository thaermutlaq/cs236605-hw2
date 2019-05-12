#!/usr/bin/python

import argparse
import os
import sys
import subprocess

for k in [32, 64]:
	for l in [2, 4, 6 ,8]:
		cmd = """-m hw2.experiments run-exp -n exp1_1_K{k}_L{l} 
		--seed 42 --bs-train 128 --batches 100 --epochs 100 --early-stopping 5 
		--filters-per-layer {k} --layers-per-block ${l} --pool-every ${poolEvery} 
		--hidden-dims 100""".format(k=k, l=l, poolEvery =l/2)
		subprocess.call(cmd, shell=True)



