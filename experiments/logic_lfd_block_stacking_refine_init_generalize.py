# -----------------------------------------------------------------------------
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of the LogicLfD project.
# Copyright (c) 2024 Idiap Research Institute <contact@idiap.ch>
# Contributor: Yan Zhang <yan.zhang@idiap.ch>
# -----------------------------------------------------------------------------

"""
This scripts illustrates how Logic-LfD is used to solve 
the block stacking problem with different initial states
"""
import os
import time
import pickle
import numpy as np
from config import BLOCK_URDF, OBJECT_URDF, STATIC_PATH, \
    PANDA_URDF, absjoin, EXP_PATH, TRAINED_MODEL_PATH
from pddlstream.algorithms.meta import solve
from pddlstream.language.constants import print_solution, PDDLProblem
from pddlstream.language.generator import from_gen_fn
from pddlstream.utils import read, INF

from examples.pybullet.utils.pybullet_tools.utils import \
    LockRenderer, HideOutput

from experiments.blocks_world_refine.primitives import \
    get_grasp_gen, get_stable_gen, get_stack_gen

from experiments.blocks_world_refine.env_panda_stacking import PandaBlockWorld

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

def generate_multi_goals(states, robot_id=0, table_id=1,
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
					*([("clear", i) for i in range(start_id, num_blocks + start_id)]))
	goals.append(goal1)

	# Intermediate Goals: Stacking blocks in a specific order
	for i in range(num_blocks, 1, -1):
		intermediate_goal = ("and", *[("on-table", k, table_id) for k in range(start_id, i)],
									*[("clear", k) for k in range(start_id, i)],
									("clear", i),
									*[("on-block", j, j + 1) for j in range(i, num_blocks + 1)],
									*[("not", ("clear", j+1)) for j in range(i, num_blocks + 1)],
									)
		intermediate_goal +=(("on-table", end_id, table_id),) 
		goals.append(intermediate_goal)

	return goals

def solve_template_tamp_problem(robots_info, tables_info, 
                                blocks_info, goal=None):
	"""
	This function generate a plan for solving the template TAMP problem
	"""
	with HideOutput():
		env = PandaBlockWorld(use_gui=USE_GUI)
		init_table_states = env.load_table(tables_info)
		init_robot_states = env.load_robot(robots_info)
		init_block_states = env.template_load_objects(blocks_info)
		init = init_robot_states + init_table_states + init_block_states

	statics, fluents = env.get_logical_state(robots_info=robots_info,
										  tables_info=tables_info,
										  blocks_info=blocks_info)
	init += statics + fluents
	problem = pddlstream_from_problem(robots_info, tables_info, 
										blocks_info, init, goal=goal)
	if VERBOSE:
		print("Template problem init: ", problem.init)
		print("Template problem goal: ", problem.goal)


	with LockRenderer(lock=False):
		init_t = time.time()
		solution = solve(problem, success_cost=INF, unit_costs=True,
						debug=DEBUG, verbose=VERBOSE)
		cost_t = time.time() - init_t
	
	# _, _, _ = print_solution(solution)
	print(f"\n Time cost for solving the template problem: {cost_t:.3f} s")

	plan, _, _ = solution
	if plan is None:
		env.disconnect()

	# Visualization
	if USE_GUI:
		# Visualization
		env.reset(robots_info=robots_info, blocks_info=blocks_info)
		env.postprocess_plan(plan)
		time.sleep(1)
	env.disconnect()

	return plan

def solve_multi_goal_tamp_problem(robots_info, tables_info, 
                                blocks_info, init_plan=None,
                                skills=None):
	"""
	This function solves the multi-goal TAMP problem
	"""
	with HideOutput():
		env = PandaBlockWorld(use_gui=USE_GUI)
		init_table_states = env.load_table(tables_info)
		init_robot_states = env.load_robot(robots_info)
		init_block_states = env.random_load_objects(blocks_info,
			max_stack_num=4)
	init = init_robot_states + init_table_states + init_block_states
	statics, fluents = env.get_logical_state(robots_info=robots_info,
										  tables_info=tables_info,
										  blocks_info=blocks_info)
	init += statics + fluents

	goals = generate_multi_goals(states=None,
								robot_id=robots_info["panda"]["id"],
								table_id=tables_info["floor"]["id"], 
								blocks_info=blocks_info)
	multi_goal = ("or", *goals)

	one_goal = goals[-1]
	one_problem = pddlstream_from_problem(robots_info, tables_info,
										blocks_info, init, goal=one_goal)
	# print("One-goal problem init: ", one_problem.init)
	# print("One-goal problem goal: ", one_problem.goal)

	with LockRenderer(lock=False):
		one_init_t = time.time()
		one_goal_solution = solve(one_problem, success_cost=INF, unit_costs=True,
						debug=DEBUG, verbose=VERBOSE 
						)
		one_time_cost = time.time() - one_init_t
	
	# print(f"\none-goal solution: ")
	# print_solution(one_goal_solution)
	# cur_plan, _, _ = one_goal_solution
	# if cur_plan is None:
	# 	env.disconnect()
		
	# # Visualization
	# if USE_GUI:
	# 	env.reset(robots_info=robots_info, blocks_info=blocks_info)
	# 	env.postprocess_plan(cur_plan)
	# 	time.sleep(1)

	"""solve multi goal problem"""
	multi_problem = pddlstream_from_problem(robots_info, tables_info, 
										blocks_info, init, goal=multi_goal)
	# print("Multi-goal problem init: ", multi_problem.init)
	# print("Multi-goal problem goal: ", multi_problem.goal)
	with LockRenderer(lock=False):
		multi_init_t = time.time()
		multi_goal_solution = solve(multi_problem, success_cost=INF, unit_costs=True,
						debug=DEBUG, verbose=VERBOSE
						)
		multi_time_cost = time.time() - multi_init_t
	
	# print(f"\nmulti-goal solution: ")
	# print_solution(multi_goal_solution)

	cur_plan, _, _ = multi_goal_solution
	if cur_plan is None:
		env.disconnect()
		
	# Visualization
	elif USE_GUI:
		env.reset(robots_info=robots_info, blocks_info=blocks_info)
		env.postprocess_plan(cur_plan)
		time.sleep(1)
	
		# goal_idx indicates which goal is achieved
		_, curr_state = env.get_logical_state(robots_info=robots_info,
										  tables_info=tables_info,
										  blocks_info=blocks_info)
		flag, goal_idx = env.find_logical_states(curr_state, goals)
		if CONNECT_PLAN:
			if flag is False:
				raise ValueError("The current state is not a goal state")
			else:
				new_plan = init_plan[int(goal_idx*2):]
				env.visualize_concatenated_plan(new_plan, skills)
		env.disconnect()
	else:
		env.disconnect()
	return one_time_cost, multi_time_cost

 
if __name__ == "__main__":
	"""Generate the action plan for template problem"""
	DEBUG = False; VERBOSE = False # By default, not log info
	USE_GUI = True; np.random.seed(0)

	# If true, connect the partial plan with demo for reaching task goal
	CONNECT_PLAN = True 

	if CONNECT_PLAN:
		"""Load Action Primitives"""
		skills = {
			'pick': None,
			'place': None,
			'stack': None,
			'unstack': None
		}

		for name in skills.keys():
			with open(absjoin(TRAINED_MODEL_PATH, f"reach_lqt_cp_dmp.pkl"), 
				"rb") as f:
				skills[name] = pickle.load(f)

	# set blocks info
	num_blocks = 4
	letters = ["a", "b", "c", "d", "e", "f", "p", "l"]
	block_names = ["cube_"+letter for letter in letters]

	# set robots info 
	robots_info = {"panda": {
			"urdf": PANDA_URDF,
			"conf": [0, 0, 0, -1.5, 0, 1.5, 0.717, 0.06, 0.06]}}

	# set tables info
	tables_info = {"floor": {
			"urdf": absjoin(OBJECT_URDF, 'short_floor.urdf'),
			"pose": [0.0, 0.0, 0.0]}}

	blocks_info = {}
	for block_name in block_names[:num_blocks]:
		blocks_info[block_name] = {
			"urdf": absjoin(BLOCK_URDF, block_name+".urdf"),
		}

	init_plan = solve_template_tamp_problem(robots_info, tables_info,
											blocks_info, goal=None)

	"""Solve the multi-goal problem"""
	info = {
		"n_itr": 100,
		"one_time_cost": [],
		"multi_time_cost": [],
		"plan_length": [],
		"plan_cost": [],
		"evaluations": [],
	}
	for i in range(info["n_itr"]):
		one_cost, multi_cost = solve_multi_goal_tamp_problem(robots_info, 
				tables_info, blocks_info, init_plan=init_plan, skills=skills)
		info["one_time_cost"].append(one_cost)
		info["multi_time_cost"].append(multi_cost)

	print(f"Average time cost for solving the multi-goal problem: {np.mean(info['multi_time_cost']):.3f} s \
		with std: {np.std(info['multi_time_cost']):.6f} s")
	print(f"Average time cost for solving the one-goal problem: {np.mean(info['one_time_cost']):.3f} s \
		with std: {np.std(info['one_time_cost']):.6f} s")   

	# folder_path = absjoin(STATIC_PATH,"time_costs", 
	# 				   "logic_dmp", "block_stacking")
	# if not os.path.exists(folder_path):
	# 	os.makedirs(folder_path)
	# np.savetxt(absjoin(folder_path, "multi_time_cost.txt"), 
	# 		np.asarray(info['multi_time_cost']))
	# np.savetxt(absjoin(folder_path, "one_time_cost.txt"), 
	# 		np.asarray(info['one_time_cost']))
	