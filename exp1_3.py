#!/usr/bin/python

import argparse
import os
import sys
import subprocess
from hw2 import experiments

for l in [2, 4, 6 ,8]:
	experiments.run_experiment(run_name = "exp1_3_L{l}_K64-128-256".format(l = l), out_dir='./results', seed=None,
                   # Training params
                   bs_train=128, bs_test=None, batches=100, epochs=120,
                   early_stopping=7, checkpoints=None, lr=0.005, reg=1e-3,
                   # Model params
                   filters_per_layer=[54, 128, 256], layers_per_block=l, pool_every=l/2,
                   hidden_dims=[100], ycn=False)



