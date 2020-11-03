import random

from agents import Wall
from assignment.custom_agents import MyAgent, MyAgentAction
from assignment.environment import Percept
from assignment.environment.things import Food, Guard


class MyReflexAgentForAlgo(MyAgent):
    pass


def MyReflexAgentForAlgoProgram(sol):
    def program(percept: 'Percept'):
        return rule_match(percept)

    def rule_match(percept) -> 'MyAgentAction':
        """Returns an action based on it's percepts"""

        for thing in percept.things:
            if isinstance(thing, Food):
                return MyAgentAction.CAN_GRAB
            elif isinstance(thing, Guard):
                return MyAgentAction.CAN_STAB
        possible_loc = [tuple(percept.front_loc), tuple(percept.back_loc), tuple(percept.right_loc),
                        tuple(percept.left_loc)]

        for point in possible_loc:
            if point in sol:
                list(point)
                break

        if percept.right_loc == list(point):
            return MyAgentAction.TURN_RIGHT
        elif percept.left_loc == list(point):
            return MyAgentAction.TURN_LEFT
        elif percept.front_loc == list(point):
            sol.remove(point)
            return MyAgentAction.MOVE_FORWARD
        elif percept.back_loc == list(point):
            return MyAgentAction.MOVE_BACKWARD

    return program
