import random

from agents import Wall
from assignment.custom_agents import MyAgent, MyAgentAction
from assignment.environment import Percept
from assignment.environment.things import Food, Guard


class MyReflexAgent(MyAgent):
    pass


def ReflexAgentProgram():
    def program(percept: 'Percept'):
        return rule_match(percept)

    def rule_match(percept) -> 'MyAgentAction':
        """Returns an action based on it's percepts"""

        for thing in percept.things:
            if isinstance(thing, Food):
                return MyAgentAction.CAN_GRAB
            elif isinstance(thing, Guard):
                return MyAgentAction.CAN_STAB

        if no_wall_among(percept.things_at_front):
            choice = random.choice((1, 2, 3, 4, 5, 6))
        else:
            choice = random.choice([1, 2])

        if choice == 1:
            return MyAgentAction.TURN_RIGHT
        elif choice == 2:
            return MyAgentAction.TURN_LEFT
        elif choice == 3:
            return MyAgentAction.MOVE_BACKWARD
        else:
            return MyAgentAction.MOVE_FORWARD

    def no_wall_among(things):
        return not bool([thing for thing in things if thing.__class__.__name__ == 'Wall'])

    return program
