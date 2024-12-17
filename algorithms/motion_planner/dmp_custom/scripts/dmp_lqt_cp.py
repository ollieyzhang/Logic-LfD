# -----------------------------------------------------------------------------
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of the RDF project.
# Copyright (c) 2024 Idiap Research Institute <contact@idiap.ch>
# Contributor: Yan Zhang <yan.zhang@idiap.ch>
# -----------------------------------------------------------------------------

import numpy as np
from copy import deepcopy
from math import factorial
from scipy.interpolate import interp1d
from scipy.linalg import block_diag

class DMP_LQT_CP():
	def __init__(self, dt=1e-2, nbData=100, nbFct=9, nbVarU=2, 
					nbDeriv=3, Qtrack=1e-5, Qreach=1, Qvia=0,
					r=1e-9, basisName="RBF"):
		
		self.dt = dt  # Time step length
		self.nbData = nbData  # Number of datapoints
		# self.nbSamples = 10  # Number of generated trajectory samples
		self.nbFct = nbFct  # Number of basis function
		self.nbVarU = nbVarU  # Control space dimension (dx1,dx2,dx3)
		self.nbDeriv = nbDeriv  # Number of static and dynamic features (nbDeriv=2 for [x,dx] and u=ddx)
		self.nbVar = self.nbVarU * self.nbDeriv  # Dimension of state vector
		self.nbVarX = self.nbVar + 1 # Augmented state space
		self.Qtrack = Qtrack  # Tracking weight term
		self.Qreach = Qreach  # Reaching weight term
		self.Qvia = Qvia  # Via-point weight term
		self.r = r  # Control weight term
		self.basisName = basisName  # can be PIECEWISE, RBF, BERNSTEIN, FOURIER
		
		# Dynamical System settings (for augmented state space)
		# =====================================
		A1d = np.zeros(self.nbDeriv)
		for i in range(self.nbDeriv):
			A1d = A1d + np.diag(np.ones((1, self.nbDeriv - i)).flatten(), i) * self.dt ** i * 1 / factorial(i)  # Discrete 1D

		B1d = np.zeros((self.nbDeriv, 1))
		for i in range(self.nbDeriv):
			B1d[self.nbDeriv - 1 - i] = self.dt ** (i + 1) * 1 / factorial(i + 1)  # Discrete 1D

		A0 = np.kron(A1d, np.eye(self.nbVarU))  # Discrete nD
		B0 = np.kron(B1d, np.eye(self.nbVarU))  # Discrete nD

		A = np.vstack((np.hstack((A0, np.zeros((self.nbVar, 1)))), np.hstack((np.zeros((self.nbVar)), 1)).reshape(1, -1)))  # Augmented A (homogeneous)
		B = np.vstack((B0, np.zeros((1, self.nbVarU))))  # Augmented B (homogeneous)

		# Build Sx and Su transfer matrices (for augmented state space)
		Sx = np.kron(np.ones((self.nbData, 1)), np.eye(self.nbVarX, self.nbVarX))
		Su = np.zeros((self.nbVarX*self.nbData, self.nbVarU * (self.nbData-1)))  # It's maybe n-1 not sure
		M = B
		for i in range(1, self.nbData):
			Sx[i*self.nbVarX:self.nbData*self.nbVarX, :] = np.dot(Sx[i*self.nbVarX:self.nbData*self.nbVarX, :], A)
			Su[self.nbVarX*i:self.nbVarX*i+M.shape[0], 0:M.shape[1]] = M
			M = np.hstack((np.dot(A, M), B))  # [0,nb_state_var-1]

		self.A = A
		self.B = B
		self.Sx = Sx
		self.Su = Su

		self._build_basis_function()

	def _build_basis_function(self):
		functions = {
			"PIECEWISE": self.build_phi_piecewise,
			"RBF": self.build_phi_rbf,
			"BERNSTEIN": self.build_phi_bernstein,
			"FOURIER": self.build_phi_fourier
		}
		phi = functions[self.basisName](self.nbData-1, 
											self.nbFct)

		# Application of basis functions to multidimensional control commands
		self.psi = np.kron(phi, np.identity(self.nbVarU))

	# Building piecewise constant basis functions
	def build_phi_piecewise(self, nb_data, nb_fct):
		phi = np.kron( np.identity(nb_fct) , np.ones((int(np.ceil(nb_data/nb_fct)),1)) )
		return phi[:nb_data]

	# Building radial basis functions (RBFs)
	def build_phi_rbf(self, nb_data, nb_fct):
		t = np.linspace(0,1,nb_data).reshape((-1,1))
		tMu = np.linspace( t[0] , t[-1] , nb_fct )
		phi = np.exp( -1e2 * (t.T - tMu)**2 )
		return phi.T

	# Building Bernstein basis functions
	def build_phi_bernstein(self, nb_data, nb_fct):
		t = np.linspace(0,1,nb_data)
		phi = np.zeros((nb_data,nb_fct))
		for i in range(nb_fct):
			phi[:,i] = factorial(nb_fct-1) / (factorial(i) * factorial(nb_fct-1-i)) * (1-t)**(nb_fct-1-i) * t**i
		return phi

	# Building Fourier basis functions
	def build_phi_fourier(self, nb_data, nb_fct):

		t = np.linspace(0,1,nb_data).reshape((-1,1))

		# Alternative computation for real and even signal
		k = np.arange(0,nb_fct).reshape((-1,1))
		phi = np.cos( t.T * k * 2 * np.pi )
		return phi.T

	def recursive_LQR(self, Qm, Rm):
		# Least squares formulation of recursive LQR with an augmented state space and and control primitives
		PSI, Su, Sx, A, B = self.psi, self.Su, self.Sx, self.A, self.B
		W = np.linalg.inv(PSI.T @ Su.T @ Qm @ Su @ PSI + PSI.T @ Rm @ PSI) @ PSI.T @ Su.T @ Qm @ Sx
		F = PSI @ W  # F with control primitives

		# Reproduction with feedback controller on augmented state space (with CP)
		self.Ka = np.empty((self.nbData-1, self.nbVarU, self.nbVarX))
		self.Ka[0, :, :] = F[0:self.nbVarU, :]
		P = np.identity(self.nbVarX)
		for t in range(self.nbData-2):
			id = np.arange((t+1)*self.nbVarU, (t+2)*self.nbVarU, step=1, dtype=int)
			P = P @ np.linalg.pinv(A - B @ self.Ka[t, :, :])
			self.Ka[t+1, :, :] = F[id, :] @ P

	def compute_reference(self, x):
		f_pos = interp1d(np.linspace(0, np.size(x, 1)-1, np.size(x, 1), dtype=int), x, kind='cubic')
		MuPos = f_pos(np.linspace(0, np.size(x, 1)-1, self.nbData))  # Position
		MuVel = np.gradient(MuPos)[1] / self.dt
		MuAcc = np.gradient(MuVel)[1] / self.dt
		# Position, velocity and acceleration profiles as references
		self.Mu = np.vstack((MuPos, MuVel, MuAcc, 
						np.zeros((self.nbVar-3*self.nbVarU, 
									self.nbData))))
		return self.Mu

	def compute_weights(self, Mu):
		"""
		Args
			Mu: reference pos, vel, acc trajectory
		"""
		# Task setting (tracking of acceleration profile and reaching of an end-point)
		Q = np.kron(np.identity(self.nbData), 
					np.diag(np.concatenate((np.zeros((self.nbVarU*2)), 
											np.ones(self.nbVarU)*self.Qtrack))))
		Q[-1-self.nbVar+1:-1-self.nbVar+2*self.nbVarU+1, -1-self.nbVar+1:-1-self.nbVar+2*self.nbVarU+1] = np.identity(2*self.nbVarU) * self.Qreach

		# Weighting matrices in augmented state form
		Qm = np.zeros((self.nbVarX*self.nbData, self.nbVarX*self.nbData))
		for t in range(self.nbData):
			id0 = np.linspace(0, self.nbVar-1, 
								self.nbVar, dtype=int) + t * self.nbVar
			id = np.linspace(0, self.nbVarX-1, self.nbVarX, dtype=int) + t * self.nbVarX
			Qm[id[0]:id[-1]+1, id[0]:id[-1]+1] = np.vstack((np.hstack((np.identity(self.nbVar), 
																		np.zeros((self.nbVar, 1)))), 
																		np.append(-Mu[:, t].reshape(1, -1), 1))) \
			@ block_diag((Q[id0[0]:id0[-1]+1, id0[0]:id0[-1]+1]), 1) @ np.vstack((np.hstack((np.identity(self.nbVar), 
																								-Mu[:, t].reshape(-1, 1))), 
																								np.append(np.zeros((1, self.nbVar)), 
																										1)))

		Rm = np.identity((self.nbData-1)*self.nbVarU) * self.r

		return Qm, Rm

	def compute_weights_viapoints(self, ref, via_points):
		"""
		Args
			ref: reference pos, vel, acc trajectory
			via_points: info of via points
		"""
		Mu = deepcopy(ref)
		# Task setting (tracking of acceleration profile, via-points, and reaching of an end-point)
		Q = np.kron(np.identity(self.nbData), 
					np.diag(np.concatenate((np.zeros((self.nbVarU*2)), 
											np.ones(self.nbVarU)*self.Qtrack))))
		Q[-1-self.nbVar+1:-1-self.nbVar+2*self.nbVarU+1, -1-self.nbVar+1:-1-self.nbVar+2*self.nbVarU+1] = np.identity(2*self.nbVarU) * self.Qreach
		for via in via_points:
			length_vp = len(via[1:])
			init_id = via[0]*self.nbVar
			end_id = init_id + length_vp
			Q[init_id:end_id, init_id:end_id] = np.identity(length_vp) * self.Qvia
			Mu[:length_vp, via[0]] = np.asarray(via[1:])
		
		# Weighting matrices in augmented state form
		Qm = np.zeros((self.nbVarX*self.nbData, self.nbVarX*self.nbData))
		for t in range(self.nbData):
			id0 = np.linspace(0, self.nbVar-1, 
								self.nbVar, dtype=int) + t * self.nbVar
			id = np.linspace(0, self.nbVarX-1, self.nbVarX, dtype=int) + t * self.nbVarX
		
			Qm[id[0]:id[-1]+1, id[0]:id[-1]+1] = np.vstack((np.hstack((np.identity(self.nbVar), 
																		np.zeros((self.nbVar, 1)))), 
																		np.append(-Mu[:, t].reshape(1, -1), 1))) \
			@ block_diag((Q[id0[0]:id0[-1]+1, id0[0]:id0[-1]+1]), 1) @ np.vstack((np.hstack((np.identity(self.nbVar), 
																								-Mu[:, t].reshape(-1, 1))), 
																								np.append(np.zeros((1, self.nbVar)), 
																										1)))

		
		
		# control command weights
		Rm = np.identity((self.nbData-1)*self.nbVarU) * self.r

		return Qm, Rm

	def step(self, t, x):
		"""
		Args:
			t - time step
			x - current augmented system state
		"""
		# Feedback control on augmented state (resulting in feedback and feedforward terms on state)
		u = -self.Ka[t,:,:] @ x
		x = self.A @ x + self.B @ u  # Update of state vector

		return x.flatten()  # State

	def generalize_traj(self, x0):
		"""
		Args:
			x0: init position
		"""
		r = np.empty((self.nbVarX, self.nbData-1))
		rx = np.append(np.append(x0, 
								np.zeros(self.nbVar-3)), 
								1).reshape(-1, 1)
		for t in range(self.nbData-1):
			rx = self.step(t, rx)
			r[:, t] = rx.flatten()
		return r