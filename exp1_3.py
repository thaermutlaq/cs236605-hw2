#!/usr/bin/python

import argparse
import os
import sys
import subprocess
from hw2 import experiments

for l in [1, 2, 3 ,4]:
	pool_every = 2 if l<=2 else 4   
	experiments.run_experiment(run_name = "exp1_3_L{l}_K64-128-256".format(l = l), out_dir='./results2', seed=None,
                   # Training params
                   bs_train=128, bs_test=None, batches=1000, epochs=200,
                   early_stopping=10, checkpoints=None, lr=0.001, reg=1e-3,
                   # Model params
                   filters_per_layer=[64, 128, 256], layers_per_block=l, pool_every=pool_every,
                   hidden_dims=[100], ycn=False)



