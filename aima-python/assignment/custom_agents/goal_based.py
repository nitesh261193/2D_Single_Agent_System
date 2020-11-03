from typing import Callable, List

from assignment.custom_agents import MyAgentAction
from assignment.custom_agents.model import MyModalBasedAgent, Model
from assignment.environment import Percept
from assignment.environment.things import Food, Guard


class MyGoalBasedAgent(MyModalBasedAgent):
    pass


class Goal:
    def __init__(self, goal_loc: 'List'):
        self.goal_loc = goal_loc

    def distance(self, location: 'List') -> int:
        return abs(location[0] - self.goal_loc[0]) + abs(location[1] - self.goal_loc[1])


def GoalBasedAgentProgram(update_state: 'Callable', model: 'Model', goal: 'Goal'):
    def program(percept: 'Percept'):
        program.last_percept = update_state(program.last_percept, program.last_action, percept, model)
        action = rule_match(program.last_percept, model, goal)
        program.last_action = action
        return action

    program.last_percept = program.last_action = None

    def rule_match(percept: Percept, model: 'Model', goal: 'Goal') -> 'MyAgentAction':
        """Returns an action based on it's percepts"""
        for thing in percept.things:
            if isinstance(thing, Food):
                return MyAgentAction.CAN_GRAB
            elif isinstance(thing, Guard):
                return MyAgentAction.CAN_STAB

        decison_dict = dict()
        if no_wall_among(percept.things_at_front):
            decison_dict[goal.distance(percept.front_loc)] = MyAgentAction.MOVE_FORWARD
        if no_wall_among(percept.things_at_right):
            distance = goal.distance(percept.right_loc)
            if distance not in decison_dict:
                decison_dict[goal.distance(percept.right_loc)] = MyAgentAction.TURN_RIGHT
        if no_wall_among(percept.things_at_left):
            distance = goal.distance(percept.left_loc)
            if distance not in decison_dict:
                decison_dict[goal.distance(percept.left_loc)] = MyAgentAction.TURN_LEFT

        distances = sorted(list(decison_dict.keys()))
        if not distances:
            return MyAgentAction.TURN_RIGHT

        return decison_dict[distances[0]]

    def no_wall_among(things):
        return not bool([thing for thing in things if thing.__class__.__name__ == 'Wall'])

    return program
