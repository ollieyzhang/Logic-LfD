# -----------------------------------------------------------------------------
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of the LogicLfD project.
# Copyright (c) 2024 Idiap Research Institute <contact@idiap.ch>
# Contributor: Yan Zhang <yan.zhang@idiap.ch>
# -----------------------------------------------------------------------------

import os
import time
import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy
from pathlib import Path
from algorithms.motion_planner.dmp_custom.scripts.dmp_lqt_cp import DMP_LQT_CP

"""Define task parameters"""
param = lambda: None # Lazy way to define an empty class in python
param.dt = 1e-2  # Time step length
param.nbData = 100  # Number of datapoints
# param.nbSamples = 10  # Number of generated trajectory samples
param.nbFct = 9  # Number of basis function
param.nbVarU = 2  # Control space dimension (dx1,dx2,dx3)
param.nbDeriv = 3  # Number of static and dynamic features (nbDeriv=2 for [x,dx] and u=ddx)
param.nbVar = param.nbVarU * param.nbDeriv  # Dimension of state vector
param.nbVarX = param.nbVar + 1 # Augmented state space
param.r = 1e-9  # Control weight term
param.basisName = "RBF"  # can be PIECEWISE, RBF, BERNSTEIN, FOURIER

lqt_cp_dmp = DMP_LQT_CP(param.dt, param.nbData, param.nbFct, param.nbVarU,
                        param.nbDeriv, param.r, param.basisName) 
"""Load data"""
# =====================================
FILE_PATH = os.path.dirname(os.path.realpath(__file__))
DATASET_LETTER_FILE = "../tests/data/letters/S.npy"
x = np.load(str(Path(FILE_PATH, DATASET_LETTER_FILE)))[0,:,:2].T

"""Compute the weight matrices"""
Mu = lqt_cp_dmp.compute_reference(x)
Qm, Rm = lqt_cp_dmp.compute_weights(Mu)

# Calculate the feedback gain
init_t = time.time()
lqt_cp_dmp.recursive_LQR(Qm, Rm)
print(f"Time taken for LQT: {time.time() - init_t}")

"""Step forward for real-time execution"""
r = np.empty((2, param.nbVarX, param.nbData-1))
# Simulated noise on state
xn = np.vstack((np.array([[3], [-3]]), np.zeros((param.nbVarX-param.nbVarU, 1))))  

time_list = []
for n in range(2):
    # initial augmented state with disturbance
    x = np.append(Mu[:, 0] + np.append(np.array([5, 4]), 
                                    np.zeros(param.nbVar-2)), 
                                    1).reshape(-1, 1)
    for t in range(param.nbData-1):
        init_t = time.time()
        x = lqt_cp_dmp.step(t, x)
        cost_t = time.time() - init_t
        time_list.append(deepcopy(cost_t))
        r[n, :, t] = x.flatten()

print(f"Average time taken for one time step: \
        {np.mean(np.asarray(time_list))}")

# Plot 2D
plt.figure()
plt.axis("on")
plt.gca().set_aspect('equal', adjustable='box')

plt.plot(Mu[0, :], Mu[1, :], c='blue', linestyle='-', linewidth=2, label="reference")
plt.scatter(Mu[0, -1], Mu[1, -1], c='red', s=100)
plt.scatter(r[0, 0, 0], r[0, 1, 0], c='black', s=50)
plt.plot(r[0, 0, :], r[0, 1, :], c='black', linestyle=':', linewidth=2, label="reproduction w/o disturbance")
plt.plot(r[1, 0, :], r[1, 1, :], c='black', linestyle='-', linewidth=2, label="reproduction w disturbance")

plt.show()
