
from config import PROJECT_DIR, TEST_PATH, absjoin
from operator import itemgetter
from copy import deepcopy

# Load PDDL related functions from LGP package
from scripts.logic.parser import PDDLParser
from scripts.logic.action import Action
from scripts.utils.helpers import frozenset_of_tuples

class PDDLHelper():
	def __init__(self, domain_path):
		self.domain = PDDLParser.parse_domain(domain_path)
	
	def applicable(self, state, positive, negative):
		# positive = LogicPlanner.match_any(state, positive)  # uncomment if ?* operator is in preconditions
		# negative = LogicPlanner.match_any(state, negative)
		return positive.issubset(state) and negative.isdisjoint(state)

	def apply(self, state, positive, negative):
		# only match any ?* for negative effects
		negative = self.match_any(state, negative)
		return state.difference(negative).union(positive)

	def match_any(self, state, group):
		for p in group:
			if '?*' in p:
				checks = [i for i, v in enumerate(p) if v != '?*']
				for state_p in state:
					if p[0] in state_p:
						p_check = ''.join(itemgetter(*checks)(p))
						state_p_check = ''.join(itemgetter(*checks)(state_p))
						if p_check == state_p_check:
							group = group.difference(frozenset([p])).union(frozenset([state_p]))
							break
		return group

	def parse_state(self, state):
		return frozenset_of_tuples(state)

	def parse_goal(self, goal):
		pos_goals, neg_goals = PDDLParser.parse_goal(goal)
		positive_goals = [frozenset_of_tuples(goal) for goal in pos_goals]
		negative_goals = [frozenset_of_tuples(goal) for goal in neg_goals]
		return positive_goals, negative_goals

	def ground_action(self, act_name, act_args):
		name, args = act_name, act_args
		act = self.domain.actions[name]
		# ground the action with the arguments
		variables = act.get_variables()
		assignment_map = dict(zip(variables, args))
		positive_preconditions = Action.replace(act.positive_preconditions, 
												assignment_map)
		negative_preconditions = Action.replace(act.negative_preconditions, 
												assignment_map)
		add_effects = Action.replace(act.add_effects, assignment_map)
		del_effects = Action.replace(act.del_effects, assignment_map)

		grounded_act = Action(name=name, 
							parameters=args,
							positive_preconditions=positive_preconditions,
							negative_preconditions=negative_preconditions,
							add_effects=add_effects, 
							del_effects=del_effects,)
		return grounded_act

	def step_plan(self, init_state, plan, verbose=False):
		domain = self.domain
		state = self.parse_state(init_state)

		step = 0
		states = []
		states.append(state)
		
		if verbose:
			print(f"\nStep: {step}, \nState: {state}")
		
		for action in plan:
			name, args = action.name, action.args
			act = domain.actions[name]

			# ground the action with the arguments
			variables = act.get_variables()
			assignment_map = dict(zip(variables, args))
			positive_preconditions = Action.replace(act.positive_preconditions, 
													assignment_map)
			negative_preconditions = Action.replace(act.negative_preconditions, 
													assignment_map)
			add_effects = Action.replace(act.add_effects, assignment_map)
			del_effects = Action.replace(act.del_effects, assignment_map)

			grounded_act = Action(name=name, 
								parameters=args,
								positive_preconditions=positive_preconditions,
								negative_preconditions=negative_preconditions,
								add_effects=add_effects, 
								del_effects=del_effects,)
			
			new_state = self.apply(state,
							grounded_act.add_effects,
							grounded_act.del_effects)
			state = new_state
			step += 1
			states.append(deepcopy(state))
			if verbose:
				print(f"\nStep: {step}," + 
					f"\nAction: {grounded_act}," +
					f"\nState: {state}")
		return states
