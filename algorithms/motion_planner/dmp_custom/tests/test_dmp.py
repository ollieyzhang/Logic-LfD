# -----------------------------------------------------------------------------
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of the LogicLfD project.
# Copyright (c) 2024 Idiap Research Institute <contact@idiap.ch>
# Contributor: Yan Zhang <yan.zhang@idiap.ch>
# -----------------------------------------------------------------------------

"""
This file compare the performance of DMP and LQT_cp
for tracking a letter in terms of trajectory similiarity,
time for generating new trajectory, 
"""

import numpy as np
import matplotlib.pyplot as plt
from config import absjoin, EXP_PATH, PROJECT_DIR
from copy import deepcopy
from algorithms.motion_planner.dmp_custom.scripts.dmp_lqt_cp import DMP_LQT_CP
from pydmps.dmp_discrete import DMPs_discrete

"""Set parameters"""
# Standard DMP parameters
n_dmps=2; n_bfs=100; ay=np.ones(2) * 10.0
# LQT_CP parameters
dt=1e-2; nbData=100; nbFct=15; nbVarU=2
nbDeriv=3; r=1e-9; basisName="RBF"

"""Load data"""
# 2D S letter
filename = absjoin(PROJECT_DIR, 
                'statistics', '2Dletters', 'S.npy')
demos = np.load(filename)[0, :, :nbVarU].T
x = demos.copy()

"""Define DMP"""
# tranin LQT_CP DMP
lqt_cp_dmp = DMP_LQT_CP(nbVarU=2)
Mu = lqt_cp_dmp.compute_reference(x)
Qm, Rm = lqt_cp_dmp.compute_weights(Mu)
# Calculate the feedback gain
lqt_cp_dmp.recursive_LQR(Qm, Rm)

# train standard DMP
dmp = DMPs_discrete(n_dmps=n_dmps, 
                    n_bfs=n_bfs, 
                    ay=ay)
dmp.imitate_path_custom(ref=Mu)
dmp.reset_state()

"""Test"""
lqt_r = np.empty((lqt_cp_dmp.nbVarX, lqt_cp_dmp.nbData-1))
dmp_r = np.empty((6, lqt_cp_dmp.nbData-1))
# initial augmented state with disturbance
rx = np.append(Mu[:, 0] + np.append(np.array([1, 1]), 
                np.zeros(lqt_cp_dmp.nbVar-2)), 
                1).reshape(-1, 1)
lqt_rx = rx.copy()
dmp_rx = rx.copy()[:6, :].flatten()
for t in range(lqt_cp_dmp.nbData-1):
    lqt_rx = lqt_cp_dmp.step(t, lqt_rx)
    lqt_r[:, t] = lqt_rx.flatten()

    dmp_rx = dmp.step_real_time(t, dmp_rx)
    dmp_r[:, t] = dmp_rx.flatten()


"""Plot"""
fig = plt.figure()
fig.set_size_inches(8, 8)
plt.suptitle("LQT_CP Encoding S Trajectory")
plt.plot(Mu[0, :], Mu[1, :], label="ref", color="black", linewidth=4)
plt.scatter(Mu[0, 0], Mu[1, 0], color="green", s=100)
plt.scatter(Mu[0, -1], Mu[1, -1], color="red", s=100)
plt.plot(lqt_r[0, :], lqt_r[1, :], label="LQT_CP", color="grey", linewidth=2)
plt.scatter(lqt_r[0, 0], lqt_r[1, 0], color="green", s=100)
plt.scatter(lqt_r[0, -1], lqt_r[1, -1], color="red", s=100)

plt.xlabel("X")
plt.ylabel("Y")
plt.tight_layout()
plt.legend()

fig = plt.figure()
fig.set_size_inches(8, 8)
plt.suptitle("DMP Encoding S Trajectory")
plt.plot(Mu[0, :], Mu[1, :], label="ref", color="black", linewidth=4)
plt.scatter(Mu[0, 0], Mu[1, 0], color="green", s=100)
plt.scatter(Mu[0, -1], Mu[1, -1], color="red", s=100)
plt.plot(dmp_r[0, :], dmp_r[1, :], label="DMP", color="grey", linewidth=2)
plt.scatter(dmp_r[0, 0], dmp_r[1, 0], color="green", s=100)
plt.scatter(dmp_r[0, -1], dmp_r[1, -1], color="red", s=100)

plt.xlabel("X")
plt.ylabel("Y")
plt.tight_layout()
plt.legend()
plt.show()