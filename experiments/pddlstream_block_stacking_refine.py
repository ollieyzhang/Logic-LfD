# -----------------------------------------------------------------------------
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of the LogicLfD project.
# Copyright (c) 2024 Idiap Research Institute <contact@idiap.ch>
# Contributor: Yan Zhang <yan.zhang@idiap.ch>
# -----------------------------------------------------------------------------

import os
import time
import pickle
import numpy as np
from copy import deepcopy
from collections import namedtuple
from config import BLOCK_URDF, OBJECT_URDF, STATIC_PATH, \
    PANDA_URDF, absjoin, EXP_PATH, TRAINED_MODEL_PATH
from pddlstream.algorithms.meta import solve
from pddlstream.language.constants import print_solution, PDDLProblem
from pddlstream.language.generator import from_gen_fn
from pddlstream.utils import read, INF
from experiments.blocks_world_refine.env_panda_stacking import PandaBlockWorld
from examples.pybullet.utils.pybullet_tools.utils import \
    LockRenderer, HideOutput
from experiments.blocks_world_refine.primitives import \
    get_grasp_gen, get_stable_gen, get_stack_gen


TABLE_HEIGHT = 0.001
BLOCK_DIMS = np.array([0.06, 0.06, 0.06])
Action = namedtuple('Action', ['name', 'args'])


def pddlstream_from_problem(robots_info, tables_info, blocks_info,
                            init, goal=None):
    
	domain_path = absjoin(EXP_PATH, 'blocks_world_refine/domain.pddl')
	stream_path = absjoin(EXP_PATH, 'blocks_world_refine/stream.pddl')
	domain_pddl = read(domain_path)
	stream_pddl = read(stream_path)
	constant_map = {}

	robot_ids = [robot_info["id"] for name, robot_info in robots_info.items()]
	table_ids = [table_info["id"] for name, table_info in tables_info.items()]
	block_ids = [block_info["id"] for name, block_info in blocks_info.items()]
	start_id = min(block_ids)
	end_id = max(block_ids)
	num_blocks = len(block_ids)

	table_id = table_ids[0]
	if goal is None:
		goal = ("and", *[("on-block", i, i + 1) for i in range(start_id, num_blocks + 1)],
						("on-table", num_blocks + 1, table_id),)
		# print("Template problem goal: ", goal)

	stream_map = {
		"find-grasp": from_gen_fn(get_grasp_gen(robot_ids[0])),
		"find-table-place": from_gen_fn(get_stable_gen(fixed=table_ids)),
		"find-block-place": from_gen_fn(get_stack_gen()),
	}

	return PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal)

def generate_multi_goals(table_id=1,
                         blocks_info=None):
	"""
	This function generate multiple goal specifications based on the plan 
	while solving the template TAMP problem
	"""
	num_blocks = len(blocks_info)
	block_ids = [block_info["id"] for name, block_info in blocks_info.items()]
	start_id = min(block_ids)
	end_id = max(block_ids)
	goals = []

	# Goal 1: All blocks on the table
	goal1 = ("and", *([("on-table", i, table_id) for i in range(start_id, num_blocks + start_id)]),
					*([("clear", i) for i in range(start_id, num_blocks + start_id)]),
					("empty", robots_info["panda"]["id"]),)
	goals.append(goal1)

	# Intermediate Goals: Stacking blocks in a specific order
	for i in range(num_blocks, 1, -1):
		intermediate_goal = ("and", *[("on-table", k, table_id) for k in range(start_id, i)],
									*[("clear", k) for k in range(start_id, i)],
									("clear", i),
									*[("on-block", j, j + 1) for j in range(i, num_blocks + 1)],
									*[("not", ("clear", j+1)) for j in range(i, num_blocks + 1)],
									)
		intermediate_goal +=(("on-table", end_id, table_id),
					   		("empty", robots_info["panda"]["id"]),) 
		goals.append(intermediate_goal)

	return goals

def load_primitives():
	"""
	Load Action Primitives"""
	skills = {
		'pick': None,
		'place': None,
		'stack': None,
		'unstack': None
	}

	for name in skills.keys():
		if name == 'pick' or 'place':
			with open(absjoin(TRAINED_MODEL_PATH, f"pick_place_lqt_cp_dmp.pkl"), 
			 "rb") as f:
				skills[name] = pickle.load(f)
		else:
			with open(absjoin(TRAINED_MODEL_PATH, f"reach_lqt_cp_dmp.pkl"), 
				"rb") as f:
				skills[name] = pickle.load(f)
	return skills

def load_world(use_gui, robots_info, tables_info, 
	blocks_info, template=True, max_stack_num=None,
	with_pose=True):
	# Load Environment
	with HideOutput():
		env = PandaBlockWorld(use_gui=use_gui)
		const_init = env.load_world(robots_info=robots_info,
			tables_info=tables_info, blocks_info=blocks_info,
			template=template, max_stack_num=max_stack_num, 
			with_pose=with_pose)
	
	statics, fluents = env.get_logical_state(robots_info=robots_info,
		tables_info=tables_info, blocks_info=blocks_info)
	init = const_init + statics + fluents
	return env, const_init, init

def generate_initial_plan():
	init_plan = [
		Action('pick', (0, 4, 1)),
		Action('stack', (0, 4, 5)),

		Action('pick', (0, 3, 1)),
		Action('stack', (0, 3, 4)),

		Action('pick', (0, 2, 1)),
		Action('stack', (0, 2, 3)),
	]
	return init_plan

def replan(env, curr_fluents, init, goals, init_plan):
	"""solve TAMP problem"""
	goal = goals[-1]
	problem = pddlstream_from_problem(robots_info, tables_info, 
		blocks_info, init, goal=goal)
	with LockRenderer(lock=True):
		solution = solve(problem, success_cost=INF, unit_costs=True,
						debug=DEBUG, verbose=VERBOSE)

	cur_plan, _, _ = solution
	if cur_plan is None:
		print("No solution found!")
	else:
		# At an unseen state, find a plan for going back
		return cur_plan[:2]


def add_disturbances(env, blocks_info, level=0):
	""" Add disturbances to the environment
	Args:
		level: 
			2: task-level disturbance, but results in a seen
				logical states in the initial plan:
				e.g. object B is placed on C before manipulating B
			3: task-level disturbance, but results in an unseen
				logical states in the initial plan:
				e.g. object A is placed on B befor manipulating B
			4: task-level disturbance, introduce new objects
				to the environment:
				e.g. object E is placed on C before manipulating B
	"""

	if level == 2:
		cube_b_id = blocks_info["cube_b"]["id"]
		cube_c_id = blocks_info["cube_c"]["id"]
		cube_d_id = blocks_info["cube_d"]["id"]

		new_b_pos = deepcopy(env.get_point(cube_d_id))
		new_c_pos = deepcopy(env.get_point(cube_d_id))

		new_b_pos[2] += BLOCK_DIMS[2] * 2
		new_c_pos[2] += BLOCK_DIMS[2]

		env.set_point(cube_b_id, new_b_pos)
		env.set_point(cube_c_id, new_c_pos)
		return None
	
	elif level == 3:
		cube_a_id = blocks_info["cube_a"]["id"]
		cube_b_id = blocks_info["cube_b"]["id"]
		new_object_pos = deepcopy(env.get_point(cube_b_id))
		new_object_pos[2] += BLOCK_DIMS[2]
		env.set_point(cube_a_id, new_object_pos)

		cube_c_id = blocks_info["cube_c"]["id"]
		cube_d_id = blocks_info["cube_d"]["id"]
		new_object_pos = deepcopy(env.get_point(cube_c_id))
		new_object_pos[2] += BLOCK_DIMS[2]
		env.set_point(cube_d_id, new_object_pos)
		return None
	
	elif level == 4:
		# add a new object to the environment
		blocks_info["cube_e"] = {
			"urdf": absjoin(BLOCK_URDF, "cube_red.urdf"),
		}
		cube_d_id = blocks_info["cube_d"]["id"]
		new_object_pos = deepcopy(env.get_point(cube_d_id))
		new_object_pos[2] += BLOCK_DIMS[2]
		blocks_info["cube_e"]["pose"] = list(new_object_pos)
		block_id, init = env.load_object(block_info=blocks_info["cube_e"])
		blocks_info["cube_e"]["id"] = block_id
		return init
	
	else:
		raise ValueError("Invalid disturbance level!")


def L1_main():
	"""Load World"""
	env, const_init, init = load_world(use_gui=USE_GUI, robots_info=robots_info,
		tables_info=tables_info, blocks_info=blocks_info, template=False,
		max_stack_num=1)


	"""Generate initial plan for the template problem"""
	goals = generate_multi_goals(blocks_info=blocks_info,
				table_id=tables_info["floor"]["id"],)
	goal = goals[-1]
	# print("\nTask goal: ", goal)
	init_plan = generate_initial_plan()

	"""Loop for reactive planning"""
	curr_plan = deepcopy(init_plan)
	init_fluents = env.get_logical_state(robots_info=robots_info,
			tables_info=tables_info, blocks_info=blocks_info)[1]
	expected_fluents = deepcopy(init_fluents)

	replan_cost_t = []
	init_t = time.time()
	for itr in range(MAX_ITER):
		# Add disturbances after execute the 1st iteration
		disturb_init = None
	
		# generate current logical states
		statics, curr_fluents = env.get_logical_state(robots_info=robots_info,
										  tables_info=tables_info,
										  blocks_info=blocks_info)
		reach_flag = env.find_logical_states(curr_fluents, goals=[goal])[0]
		
		if reach_flag is True:
			# if reach the goal, break the loop
			cost_t = time.time() - init_t
			print(f"Total time cost: {cost_t:.2f} seconds")
			env.disconnect()
			return cost_t, replan_cost_t
		else:
			init = const_init + statics + curr_fluents
			if disturb_init is not None:
				# in case new object is introduced
				init += disturb_init
			init_t_replan = time.time()
			plan = replan(env=env, curr_fluents=curr_fluents, 
				init_plan=init_plan, init=init, goals=goals)
			cost_t_replan = time.time() - init_t_replan
			replan_cost_t.append(cost_t_replan)
			# visualization
			env.visualize_concatenated_plan(plan[:2], skills=skills)
			expected_fluents = env.get_logical_state(robots_info=robots_info,
					tables_info=tables_info, blocks_info=blocks_info)[1]
	
		if itr == MAX_ITER-1:
			print("Reach the maximum iteration!")
			env.disconnect()
			break


def L2_main():
	"""Load World"""
	# plan_env, const_init, init = load_world(use_gui=False)
	env, const_init, init = load_world(use_gui=USE_GUI, robots_info=robots_info,
		tables_info=tables_info, blocks_info=blocks_info, template=False,
		max_stack_num=1)

	"""Generate initial plan for the template problem"""
	goals = generate_multi_goals(blocks_info=blocks_info,
				table_id=tables_info["floor"]["id"],)
	goal = goals[-1]
	# print("\nTask goal: ", goal)
	init_plan = generate_initial_plan()

	"""Loop for reactive planning"""
	curr_plan = deepcopy(init_plan)

	replan_cost_t = []
	init_t = time.time()
	
	# Add disturbances after execute the 1st iteration
	disturb_init = None
	
	disturb_init = add_disturbances(env, blocks_info=blocks_info, level=2)
	
	# generate current logical states
	statics, curr_fluents = env.get_logical_state(robots_info=robots_info,
										tables_info=tables_info,
										blocks_info=blocks_info)
	
	init = const_init + statics + curr_fluents
	if disturb_init is not None:
		# in case new object is introduced
		init += disturb_init
	init_t_replan = time.time()
	plan = replan(env=env, curr_fluents=curr_fluents, 
		init_plan=init_plan, init=init, goals=goals)
	cost_t_replan = time.time() - init_t_replan
	
	# visualization
	env.visualize_concatenated_plan(plan, skills=skills)

	cost_t = time.time() - init_t
	print(f"time cost {cost_t:.2f}")
	env.disconnect()
	return cost_t, cost_t_replan
		

def L3_main():
	"""Load World"""
	# plan_env, const_init, init = load_world(use_gui=False)
	env, const_init, init = load_world(use_gui=USE_GUI, robots_info=robots_info,
		tables_info=tables_info, blocks_info=blocks_info, template=False,
		max_stack_num=1)

	"""Generate initial plan for the template problem"""
	goals = generate_multi_goals(blocks_info=blocks_info,
				table_id=tables_info["floor"]["id"],)
	goal = goals[-1]
	# print("\nTask goal: ", goal)
	init_plan = generate_initial_plan()

	"""Loop for reactive planning"""
	curr_plan = deepcopy(init_plan)
	init_fluents = env.get_logical_state(robots_info=robots_info,
			tables_info=tables_info, blocks_info=blocks_info)[1]
	expected_fluents = deepcopy(init_fluents)

	replan_cost_t = []
	init_t = time.time()
	for itr in range(MAX_ITER):
		# Add disturbances after execute the 1st iteration
		disturb_init = None
		if itr == 0:
			disturb_init = add_disturbances(env, blocks_info, level=3)
		# generate current logical states
		statics, curr_fluents = env.get_logical_state(robots_info=robots_info,
										  tables_info=tables_info,
										  blocks_info=blocks_info)
		reach_flag = env.find_logical_states(curr_fluents, goals=[goal])[0]
		
		if reach_flag is True:
			# if reach the goal, break the loop
			cost_t = time.time() - init_t
			print(f"Total time cost: {cost_t:.2f} seconds")
			env.disconnect()
			return cost_t, replan_cost_t
		else:
			init = const_init + statics + curr_fluents
			if disturb_init is not None:
				# in case new object is introduced
				init += disturb_init
			init_t_replan = time.time()
			plan = replan(env=env, curr_fluents=curr_fluents, 
				init_plan=init_plan, init=init, goals=goals)
			cost_t_replan = time.time() - init_t_replan
			print(f"time cost for replanning {cost_t_replan:.4f}")
			replan_cost_t.append(cost_t_replan)
			# visualization
			env.visualize_concatenated_plan(plan[:2], skills=skills)
			expected_fluents = env.get_logical_state(robots_info=robots_info,
					tables_info=tables_info, blocks_info=blocks_info)[1]
	
		if itr == MAX_ITER-1:
			print("Reach the maximum iteration!")
			env.disconnect()
			break

def L4_main():
	"""Load World"""
	if blocks_info.get("cube_e") is not None:
		blocks_info.pop("cube_e")
	env, const_init, init = load_world(use_gui=USE_GUI, robots_info=robots_info,
		tables_info=tables_info, blocks_info=blocks_info, template=False,
		max_stack_num=1)

	"""Generate initial plan for the template problem"""
	goals = generate_multi_goals(blocks_info=blocks_info,
				table_id=tables_info["floor"]["id"],)
	goal = goals[-1]
	# print("\nTask goal: ", goal)
	init_plan = generate_initial_plan()

	"""Loop for reactive planning"""
	curr_plan = deepcopy(init_plan)
	init_fluents = env.get_logical_state(robots_info=robots_info,
			tables_info=tables_info, blocks_info=blocks_info)[1]
	expected_fluents = deepcopy(init_fluents)

	replan_cost_t = []
	init_t = time.time()
	for itr in range(MAX_ITER):
		# Add disturbances after execute the 1st iteration
		disturb_init = None
		if itr == 0:
			disturb_init = add_disturbances(env, blocks_info, level=4)
		# generate current logical states
		statics, curr_fluents = env.get_logical_state(robots_info=robots_info,
										  tables_info=tables_info,
										  blocks_info=blocks_info)
		reach_flag = env.find_logical_states(curr_fluents, goals=[goal])[0]
		
		if reach_flag is True:
			# if reach the goal, break the loop
			cost_t = time.time() - init_t
			print(f"Total time cost: {cost_t:.2f} seconds")
			env.disconnect()
			return cost_t, replan_cost_t
		else:
			init = const_init + statics + curr_fluents
			if disturb_init is not None:
				# in case new object is introduced
				init += disturb_init
			init_t_replan = time.time()
			plan = replan(env=env, curr_fluents=curr_fluents, 
				init_plan=init_plan, init=init, goals=goals)
			cost_t_replan = time.time() - init_t_replan
			replan_cost_t.append(cost_t_replan)
			# visualization
			env.visualize_concatenated_plan(plan[:2], skills=skills)
			expected_fluents = env.get_logical_state(robots_info=robots_info,
					tables_info=tables_info, blocks_info=blocks_info)[1]
	
		if itr == MAX_ITER-1:
			print("Reach the maximum iteration!")
			env.disconnect()
			break


if __name__ == '__main__':
	"""Generate the action plan for template problem"""
	DEBUG = False
	VERBOSE = False # By default, not log info
	USE_GUI = True
	CONNECT_PLAN = True
	MAX_ITER = 30
	np.random.seed(0)
	
	"""Define objects info"""
	# set blocks info
	num_blocks = 4
	letters = ["a", "b", "c", "d", "e", "f", "p", "l"]
	block_names = ["cube_"+letter for letter in letters]
	block_poses = [
		[0.3, -0.2, TABLE_HEIGHT+BLOCK_DIMS[2]/2], # A
		[0.3, 0.2, TABLE_HEIGHT+BLOCK_DIMS[2]/2], # B
		[0.45, -0.2, TABLE_HEIGHT+BLOCK_DIMS[2]/2], # C
		[0.45, 0.2, TABLE_HEIGHT+BLOCK_DIMS[2]/2], # D
		]
	# set robots info 
	robots_info = {"panda": {
			"urdf": PANDA_URDF,
			"conf": [0, 0, 0, -1.5, 0, 1.5, 0.717, 0.06, 0.06]}}

	# set tables info
	tables_info = {"floor": {
			"urdf": absjoin(OBJECT_URDF, 'short_floor.urdf'),
			"pose": [0.0, 0.0, 0.0]}}

	blocks_info = {}
	for i, block_name in enumerate(block_names[:num_blocks]):
		blocks_info[block_name] = {
			"urdf": absjoin(BLOCK_URDF, block_name+".urdf"),
			"pose": np.array(block_poses[i]),
		}

	"""Load Manipulation Primitives"""
	skills = load_primitives()
	
	levels = ["L2"]
	
	task_cost_t_list = []
	replan_cost_t_list = []
	for _ in range(10):
		task_cost_t, replan_cost_t = L3_main()
		task_cost_t_list.append(task_cost_t)
		replan_cost_t_list.append(replan_cost_t)

	# print(f"Average time cost: {np.mean(task_cost_t_list):.2f} seconds \
	# with std: {np.std(task_cost_t_list):.2f} seconds")
	# print(f"Average time cost for replanning: {np.mean(replan_cost_t_list):.6f} seconds \
	# with std: {np.std(replan_cost_t_list):.6f} seconds")
	
	# folder_path = absjoin(STATIC_PATH,"time_costs", 
	# 				"pddlstream", "block_stacking")
	# if not os.path.exists(folder_path):
	# 	os.makedirs(folder_path)
	# np.savetxt(absjoin(folder_path, "L2_task_cost_t_list.txt"), 
	# 		np.asarray(task_cost_t_list))
	# np.savetxt(absjoin(folder_path, "L2_replan_cost_t_list.txt"), 
	# 		np.asarray(replan_cost_t_list))