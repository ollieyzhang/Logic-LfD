# -----------------------------------------------------------------------------
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of the LogicLfD project.
# Copyright (c) 2024 Idiap Research Institute <contact@idiap.ch>
# Contributor: Yan Zhang <yan.zhang@idiap.ch>
# -----------------------------------------------------------------------------

import os
import time
import numpy as np
import pybullet as p
import pybullet_data
from datetime import datetime
from copy import deepcopy
from experiments.config import absjoin, EXP_PATH

from examples.pybullet.utils.pybullet_tools.utils import \
    set_pose, set_point, Pose, Point, disconnect, HideOutput,\
    set_joint_positions, get_movable_joints, get_link_pose, \
    wait_for_duration, get_point, connect, set_default_camera, \
    load_model, draw_global_system, enable_gravity

from experiments.poisson_disc_sampling import PoissonSampler, GridSampler
from experiments.blocks_world_refine.primitives import Command, BodyPose, BodyConf, \
    get_tool_link


TABLE_HEIGHT = 0.001
BLOCK_DIMS = np.array([0.06, 0.06, 0.06])
BLOCK_HIGHTS = [0.031, 0.091, 0.151, 0.211, 0.271]
R = max(BLOCK_DIMS[:2]) * np.sqrt(2)
EXTENT = np.array([0.3, 1])
CENTER = np.array([0.25, -0.5])
GRASP_DIST = 0.1
MAX_ARM_REACH = 0.7 # the actual limit is 0.855

class PandaBlockWorld():
	def __init__(self, use_gui=True):
		connect(use_gui=use_gui)
		set_default_camera(yaw=90, distance=1.5)
		draw_global_system(length=0.1)
		enable_gravity()
		
	def load_world(self, robots_info, tables_info, blocks_info,
		template=True, max_stack_num=None, with_pose=True):
		
		init_robot_states = self.load_robot(robots_info)
		init_table_states = self.load_table(tables_info)
		if template:
			if with_pose:
				init_block_states = self.template_load_objects_with_pose(blocks_info)
			else:
				init_block_states = self.template_load_objects(blocks_info, 
										table_id=tables_info["floor"]["id"])
		else:
			init_block_states = self.random_load_objects(blocks_info,
												max_stack_num=max_stack_num,)
		init = init_robot_states + init_table_states + init_block_states
		return init

	def load_robot(self, robots_info):
		init_robot_states = []
		with HideOutput():
			for name, robot_info in robots_info.items():
				robot_id = load_model(robot_info["urdf"], 
								fixed_base=True)
				robot_info["id"] = robot_id

				body_conf = BodyConf(robot_id, 
								robot_info["conf"])
			
				self.set_joint_positions(body_conf.body,
									body_conf.configuration,
									gripper=True)
				init_robot_states += [("arm", robot_id),
									("empty", robot_id),
										]
		return init_robot_states

	def load_table(self, tables_info):
		init_table_states = []
		with HideOutput():
			for name, table_info in tables_info.items():
				table_id = load_model(table_info["urdf"], 
								fixed_base=True)
				table_info["id"] = table_id
				init_table_states += [("table", table_id),]
		return init_table_states
	
	def load_object(self, block_info):
		with HideOutput():
			block_id = load_model(block_info["urdf"], 
								fixed_base=False)
			block_info["id"] = block_id
			block_pose = block_info["pose"]
			set_point(block_id, np.asarray(block_pose))
		init = [("block", block_id),]
		return block_id, init
	
	def template_load_objects_with_pose(self, blocks_info):
		init = []
		with HideOutput():
			for name, object_info in blocks_info.items():
				block_id = load_model(object_info["urdf"], 
								fixed_base=False)
				object_info["id"] = block_id
				block_pose = object_info["pose"]
				set_point(block_id, block_pose)
				init += [("block", block_id),]
		return init

	def template_load_objects(self, blocks_info):
		# This function load every objects on the table
		# use the random_load_objects with max_stack_num = 0
		init_obj_states = self.random_load_objects(blocks_info, 
													max_stack_num=1)
		return init_obj_states

	def random_load_objects(self, blocks_info, max_stack_num=None,
							):
		init_pose = Pose(Point(x=0, y=0, z=TABLE_HEIGHT+BLOCK_DIMS[2]/2))
		block_ids = []
		for name, block_info in blocks_info.items():
			block_id = load_model(block_info["urdf"])
			block_info["id"] = block_id
			set_pose(block_id, init_pose)
			block_ids.append(block_id)

		num_blocks = len(block_ids)
		sampler = GridSampler(extent=EXTENT,
						r=np.max(BLOCK_DIMS)*2.5,
						centered=False)
		
		# sample objects within robot arm's workspace
		filter = lambda point: np.linalg.norm(point) > MAX_ARM_REACH
		points = sampler.make_samples(num=num_blocks*10, filter=filter)
		
		blocks = list(blocks_info.keys())
		stacking = self.make_random_stacking(blocks, max_stack_num)
		
		init_obj_states = []
		for stack in stacking:
			# first block in the stack is on table
			point = points.pop(-1) + CENTER
			point = np.append(point, TABLE_HEIGHT+BLOCK_DIMS[2]/2)
			
			blocks_info[stack[0]]["pose"] = point.copy()
			block_pose = BodyPose(blocks_info[stack[0]]["id"], 
								Pose(Point(x=blocks_info[stack[0]]["pose"][0],
											y=blocks_info[stack[0]]["pose"][1],
											z=blocks_info[stack[0]]["pose"][2]
											)))
			set_pose(block_pose.body, block_pose.pose)

			init_obj_states += [("block", block_pose.body),]
			
			# The following blocks are stacked on the first one
			for block in stack[1:]:
				point[2] += BLOCK_DIMS[2] 
				blocks_info[block]["pose"] = point.copy()
				block_pose = BodyPose(blocks_info[block]["id"], 
								Pose(Point(x=blocks_info[block]["pose"][0],
											y=blocks_info[block]["pose"][1],
											z=blocks_info[block]["pose"][2]
											)))
				set_pose(block_pose.body, block_pose.pose)
				
				init_obj_states += [("block", block_pose.body),]
	
		return init_obj_states

	def get_logical_state(self, robots_info, tables_info, blocks_info):
		"""This function generates the logical states of current scene
		"""
		statics, fluents = [], []
		for name, robot_info in robots_info.items():
			robot_id = robot_info["id"]
			# assume the robot is always empty while generating the logical states
			fluents += [("empty", robot_id)]

		# Get object's positions
		obj_infos = [[info["id"]]+list(get_point(info["id"])) for _, info in blocks_info.items()]
		obj_infos = np.asarray(obj_infos)
		# sort object based on their hight
		sorted_idx = np.argsort(obj_infos[:, 3])
		sorted_obj_infos = obj_infos[sorted_idx, :]
		
		# Define the rules for logical states
		stack_list = []
		for obj_info in sorted_obj_infos:
			block_pose = BodyPose(int(obj_info[0]),
						Pose(np.array(obj_info[1:])) )
			statics += [("worldpose", int(obj_info[0]), block_pose),
						# this is a fluent, but ignored for logical states comparison
						("atpose", int(obj_info[0]), block_pose), 
						]
			
			for i, h in enumerate(BLOCK_HIGHTS):
				if abs(obj_info[3] - h) < 0.015:
					if i == 0:
						fluents += [("on-table", int(obj_info[0]), 
				   					tables_info["floor"]["id"]),]
						stack_list.append([obj_info])
						break
					else:
						for j, stack in enumerate(stack_list):
							if np.linalg.norm(obj_info[1:3] - stack[-1][1:3]) < 0.015:
								fluents += [("on-block", int(obj_info[0]), int(stack[-1][0])),
											("not", ("clear", int(stack[-1][0]))),]
								stack.append(obj_info)
								break

		for stack in stack_list:
			fluents += [("clear", int(stack[-1][0])),]
		# fluents contains the logical states for state comparison
		
		return statics, fluents

	def find_logical_states(self, states, goals):
		"""
		this function check if the given states is in a set of logical states
		Args:
			states: current logical states
			goals: a set of logical states
		Return:
			True if states is in goals and its index in goals
		"""
		states = ('and', *states)
		flag, idx = False, None
		for i, conjunction in enumerate(goals):
			if set(conjunction).issubset(set(states)):
				flag = True
				idx = i
				break
		return flag, idx
	
	def make_random_stacking(self, blocks, max_stack_num=None, 
								num_stacks=None):
		num_blocks = len(blocks)
		block_perm = blocks.copy()
		np.random.shuffle(block_perm)
		stacking = set()
		
		# Generate specific stacking
		if max_stack_num is not None:
			assert (
				0 < max_stack_num <= num_blocks
			), "Max stack height must be a integer greater than 0 and less than the number of blocks"
			num_max_stacks = num_blocks//max_stack_num
			num_stack = max_stack_num
			while len(block_perm) > 0:
				stacking.add(tuple(block_perm[:num_stack]))
				block_perm = block_perm[num_stack:]
				num_stack = min(np.random.randint(1, max_stack_num + 1), len(block_perm))
		# Generate random stacking
		else:
			lower_num = 0
			if num_blocks == 0:
				return stacking

			elif num_blocks == 1:
				return set([tuple(block_perm)]) | stacking

			if num_stacks is None:
				num_splits = np.random.randint(lower_num, num_blocks)
			else:
				assert num_stacks >= 0 and num_stacks < num_blocks, "Invalid stack number"
				num_splits = num_stacks - 1
			split_locs = np.random.choice(
				list(range(1, num_blocks)), size=num_splits, replace=False
			)
			split_locs.sort()
			split_locs = np.append(split_locs, num_blocks)
			i = 0
			for split_loc in split_locs:
				stacking.add(tuple(block_perm[i:split_loc]))
				i = split_loc
		return stacking

	def grasp(self, robot_id, obj_id, target_pos):
		tool_link = get_tool_link(robot_id)
		init_panda_pos, target_quat = self.get_eef_pose(robot_id)
		panda_pos = deepcopy(init_panda_pos)
		while np.linalg.norm(np.array(panda_pos) - np.array(target_pos)) > 0.01:
			conf = p.calculateInverseKinematics(robot_id,
													tool_link,
													target_pos,
													target_quat)
			self.set_joint_positions(robot_id, conf)
			wait_for_duration(0.02)
			panda_pos = self.get_eef_pose(robot_id)[0]
		
		# close the gripper
		grasp_conf = conf[:-2] + (0.03, 0.03)
		self.set_joint_positions(robot_id, grasp_conf)
		wait_for_duration(0.02)
		
		while np.linalg.norm(np.array(panda_pos) - np.array(init_panda_pos)) > 0.01:
			conf = p.calculateInverseKinematics(robot_id,
													tool_link,
													init_panda_pos,
													target_quat)
			self.set_joint_positions(robot_id, conf)
			wait_for_duration(0.02)
			panda_pos = self.get_eef_pose(robot_id)[0]
			set_point(obj_id, panda_pos)
			wait_for_duration(0.02)

	def ungrasp(self, robot_id, obj_id, target_pos):
		tool_link = get_tool_link(robot_id)
		init_panda_pos, target_quat = self.get_eef_pose(robot_id)
		panda_pos = deepcopy(init_panda_pos)
		while np.linalg.norm(np.array(panda_pos) - np.array(target_pos)) > 0.01:
			conf = p.calculateInverseKinematics(robot_id,
					tool_link, target_pos, target_quat)
			self.set_joint_positions(robot_id, conf)
			wait_for_duration(0.02)
			panda_pos = self.get_eef_pose(robot_id)[0]
			set_point(obj_id, panda_pos)
			wait_for_duration(0.02)
		
		# open the gripper
		grasp_conf = conf[:-2] + (0.06, 0.06)
		self.set_joint_positions(robot_id, grasp_conf)
		# adjust object's position for better rendering
		obj_pos = np.asarray(target_pos)
		self.set_point(obj_id, obj_pos)
		wait_for_duration(0.02)
		
		# return the initial pose
		while np.linalg.norm(np.array(panda_pos) - np.array(init_panda_pos)) > 0.01:
			conf = p.calculateInverseKinematics(robot_id,
													tool_link,
													init_panda_pos,
													target_quat)
			self.set_joint_positions(robot_id, conf)
			wait_for_duration(0.02)
			panda_pos = self.get_eef_pose(robot_id)[0]

	def postprocess_plan(self, plan):
		"""
		this function visualize the generated plan
		"""
		for name, args in plan:
			robot_id, obj_id = args[:2]
			obj_pos = args[3].value[0]
			target_pos = np.asarray(deepcopy(obj_pos))
			target_pos[2] += GRASP_DIST
			if name in ['pick', 'unstack']:
				self.state_reach(robot_id, target_pos, 
									tool=obj_id, attach=False)
				self.grasp(robot_id, obj_id, obj_pos)
			elif name in ['place', 'stack']:
				self.state_reach(robot_id, target_pos, 
									tool=obj_id, attach=True)
				self.ungrasp(robot_id, obj_id, obj_pos)
			else:
				raise NotImplementedError(name)
	
	def step_task_plan(self, action, skills):
		name, args = action.name, action.args
		robot_id, obj_id = args[:2]
		if name == 'place':
			obj_pos = args[3].value[0]
		elif name == 'stack':
			lower_obj_id = args[2]
			obj_pos = np.asarray(get_point(lower_obj_id))
			obj_pos[2] += BLOCK_DIMS[2]
		elif name in ['pick', 'unstack']:
			obj_pos = get_point(obj_id)
		else:
			raise NotImplementedError(name)
		target_pos = np.asarray(deepcopy(obj_pos))
		target_pos[2] += GRASP_DIST
		init_pos = self.get_eef_pose(robot_id)[0]
		# calculate object-centric initial position
		rel_pos = init_pos - target_pos
		# generate object centric trajectory
		oc_traj = skills[name].generalize_traj(rel_pos)[:3, :]
		# transform to world coordinate
		traj = oc_traj + target_pos[:, np.newaxis]
		if name in ['pick', 'unstack']:
			self.execute_traj(robot_id, obj=obj_id, traj=traj,
							attach=False)
			self.grasp(robot_id, obj_id, obj_pos)
		else:
			# place or stack
			self.execute_traj(robot_id, obj=obj_id, traj=traj,
							attach=True)
			self.ungrasp(robot_id, obj_id, obj_pos)

	def visualize_concatenated_plan(self, plan, skills):
		for action in plan:
			self.step_task_plan(action, skills)
			
	
	def reset(self, robots_info, blocks_info):
		for name, robot_info in robots_info.items():
			
			self.set_joint_positions(robot_info["id"],
								robot_info["conf"])
		
		for name, block_info in blocks_info.items():
			set_pose(block_info["id"], 
						Pose(block_info["pose"]))
			
	def get_movable_joints(self, robot_id, gripper=True):
		joints = get_movable_joints(robot_id)
		if gripper:
			return joints
		else:
			return joints[:-2]

	def set_joint_positions(self, robot_id, conf, gripper=True):
		joints = get_movable_joints(robot_id)
		if gripper:
			set_joint_positions(robot_id, joints, conf)
		else:
			set_joint_positions(robot_id, joints[:-2], conf)

	def get_eef_pose(self, robot_id):
		tool_link = get_tool_link(robot_id)
		return get_link_pose(robot_id, tool_link)
	
	def set_point(self, body, point):
		set_point(body, point)

	def get_point(self, body):
		return np.asarray(get_point(body))
	
	def state_reach(self, robot_id, target_pos, 
					object=None, tool=None, attach=False):
		tool_link = get_tool_link(robot_id)
		Kp = 3 / 2
		Kd = np.sqrt(2*Kp) / 2 
		dt = 0.02
		eef_pos_list = []
		panda_pos, target_quat = get_link_pose(robot_id, tool_link)
		for _ in range(1000):
			# Calculate a distance to move
			dx = target_pos[0] - panda_pos[0]   # x distance to move
			dy = target_pos[1] - panda_pos[1]   # y distance to move
			dz = target_pos[2] - panda_pos[2]   # z distance to move

			# Calculate a velocity
			pd_x = Kp * dx + Kd * dx / dt
			pd_y = Kp * dy + Kd * dy / dt
			pd_z = Kp * dz + Kd * dz / dt
			action = [pd_x, pd_y, pd_z]
			dv = 0.0022 
			dx = action[0] * dv     # panda's moving distance (x coordinate)
			dy = action[1] * dv 
			dz = action[2] * dv 
			new_panda_pos = [panda_pos[0] + dx,
						panda_pos[1] + dy,
						panda_pos[2] + dz]
			conf = p.calculateInverseKinematics(robot_id,
												tool_link,
												new_panda_pos,
												target_quat)
			self.set_joint_positions(robot_id, conf)
			panda_pos, _ = get_link_pose(robot_id, tool_link)
			if attach:
				if object is not None:
					obj_pos = get_point(object)
					new_obj_pos = [obj_pos[0] + dx,
							obj_pos[1] + dy,
							obj_pos[2]]
					set_point(object, new_obj_pos)
				if tool is not None:    
					set_point(tool, panda_pos)
			wait_for_duration(dt)
			eef_pos_list.append(panda_pos)

			# if reach the target position, break the loop
			if np.linalg.norm(np.array(panda_pos) - np.array(target_pos)) < 0.005:
				break
		return eef_pos_list
	
	def execute_traj(self, robot_id, obj, traj, attach=False):
		tool_link = get_tool_link(robot_id)
		target_quat = get_link_pose(robot_id, tool_link)[1]
		for i in range(traj.shape[1]):
			target_point = traj[:, i]
			conf = p.calculateInverseKinematics(robot_id, 
												tool_link, 
												target_point,
												target_quat)
			self.set_joint_positions(robot_id, conf)
			wait_for_duration(0.02)
			if attach:
				set_point(obj, target_point)
				# wait_for_duration(0.02)
	
	def wait_for_duration(self, duration):
		wait_for_duration(duration)

	def disconnect(self):
		disconnect()
