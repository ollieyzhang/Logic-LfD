# -----------------------------------------------------------------------------
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of the LogicLfD project.
# Copyright (c) 2024 Idiap Research Institute <contact@idiap.ch>
# Contributor: Yan Zhang <yan.zhang@idiap.ch>
# -----------------------------------------------------------------------------

# %%
import time
import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy
from .config import PROJECT_DIR, absjoin
from algorithms.motion_planner.dmp_custom.scripts.dmp_lqt_cp import DMP_LQT_CP

# %%
"""Define task parameters"""
param = lambda: None # Lazy way to define an empty class in python
param.dt = 1e-2  # Time step length
param.nbData = 100  # Number of datapoints
# param.nbSamples = 10  # Number of generated trajectory samples
param.nbFct = 10  # Number of basis function
param.nbVarU = 3  # Control space dimension (dx1,dx2,dx3)
param.nbDeriv = 3  # Number of static and dynamic features (nbDeriv=2 for [x,dx] and u=ddx)
param.nbVar = param.nbVarU * param.nbDeriv  # Dimension of state vector
param.nbVarX = param.nbVar + 1 # Augmented state space
param.r = 1e-9  # Control weight term
param.basisName = "BERNSTEIN"  # can be PIECEWISE, RBF, BERNSTEIN, FOURIER

lqt_cp_dmp = DMP_LQT_CP(param) 

"""Load Data"""
filename = absjoin(PROJECT_DIR, 
                'tests', 'data', 
                'block_stacking',
                '2023-12-14-17:30:49',
                'plan.npy')
demos = np.load(filename, allow_pickle=True,
                encoding='latin1')
# %%
# generate trajectory for motion "pick"
pick_move_traj = np.asarray(demos[0]["eef_pose"][0])[:, :3]
pick_act_traj = np.asarray(demos[1]["eef_pose"][0])[:1, :3] 
pick_traj = np.concatenate((pick_move_traj, 
                            pick_act_traj), 
                            axis=0)

# generate trajectory for motion "place"
place_traj = pick_traj.copy()[::-1, :]

# %%
# generate trajectory for motion "stack"
stack_init_traj = pick_act_traj.copy()[::-1, :]
stack_move_traj = np.asarray(demos[2]["eef_pose"][0])[:, :3]
stack_act_traj = np.asarray(demos[3]["eef_pose"][0])[:, :3]
stack_traj = np.concatenate((stack_init_traj,
                            stack_move_traj, 
                            stack_act_traj), 
                            axis=0)
# generate trajectory for motion "unstack"
unstack_traj = stack_traj.copy()[::-1, :]

""" Generate LQT_CP_DMP agent for each skills"""
x = stack_move_traj.T
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
    rx = np.append(Mu[:, 0] + np.append(np.array([-0.1, -0.1, 0.1]), 
                                    np.zeros(param.nbVar-3)), 
                                    1).reshape(-1, 1)
    for t in range(param.nbData-1):
        init_t = time.time()
        rx = lqt_cp_dmp.step(t, rx)
        cost_t = time.time() - init_t
        time_list.append(deepcopy(cost_t))
        r[n, :, t] = rx.flatten()

print(f"Average time taken for one time step: \
        {np.mean(np.asarray(time_list))}")

# plot 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(Mu[0, :], Mu[1, :], Mu[2, :], 
        c='blue', linestyle='-', linewidth=2, 
        label="reference")
ax.scatter(Mu[0, -1], Mu[1, -1], Mu[2, -1], 
           c='red', s=100)
ax.scatter(r[0, 0, 0], r[0, 1, 0], r[0, 2, 0], 
           c='black', s=50)
ax.plot(r[0, 0, :], r[0, 1, :], r[0, 2, :],
        c='black', linestyle=':', linewidth=2, 
        label="reproduction w/o disturbance")
ax.plot(r[1, 0, :], r[1, 1, :], r[1, 2, :],
        c='black', linestyle='-', linewidth=2, 
        label="reproduction w disturbance")

# Set labels for the axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.tight_layout()
plt.show()
