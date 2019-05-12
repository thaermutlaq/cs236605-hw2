#!/usr/bin/python

import argparse
import os
import sys
import subprocess
from hw2 import experiments

for k in [32, 64]:
	for l in [2, 4, 6 ,8]:
		experiments.run_experiment(run_name = "exp1_1_K{k}_L{l}".format(k = k, l = l), out_dir='./results', seed=None,
                   # Training params
                   bs_train=128, bs_test=None, batches=100, epochs=120,
                   early_stopping=10, checkpoints=None, lr=0.005, reg=1e-3,
                   # Model params
                   filters_per_layer=[k], layers_per_block=l, pool_every=l/2,
                   hidden_dims=[100], ycn=False)



