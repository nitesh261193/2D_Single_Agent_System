import random
from collections import defaultdict
from typing import Callable

from assignment.custom_agents import MyAgent, MyAgentAction
from assignment.environment import Percept
from assignment.environment.things import Food, Guard


class MyModalBasedAgent(MyAgent):
    pass


class Model:
    def __init__(self):
        self.history = defaultdict(list)
        self.visited = set()


def update(last_percept: 'Percept', last_action: 'MyAgentAction', percept: 'Percept', model: 'Model'):
    if not last_percept:
        return percept
    model.history[tuple(last_percept.agent.location)].append(last_action)
    if last_action == MyAgentAction.MOVE_FORWARD:
        model.visited.add(tuple(percept.agent.location))
    return percept


def ModelBasedAgentProgram(update_state: 'Callable', model: 'Model'):
    def program(percept: 'Percept'):
        program.last_percept = update_state(program.last_percept, program.last_action, percept, model)
        action = rule_match(program.last_percept, model)
        program.last_action = action
        return action

    program.last_percept = program.last_action = None

    def rule_match(percept: Percept, model: 'Model') -> 'MyAgentAction':
        """Returns an action based on it's percepts"""
        for thing in percept.things:
            if isinstance(thing, Food):
                return MyAgentAction.CAN_GRAB
            elif isinstance(thing, Guard):
                return MyAgentAction.CAN_STAB

        if tuple(percept.front_loc) in model.visited or wall_among(percept.things_at_front):
            choice = random.choice((1, 2))
        else:
            choice = random.choice((1, 2, 3, 4))

        if choice == 1:
            return MyAgentAction.TURN_RIGHT
        elif choice == 2:
            return MyAgentAction.TURN_LEFT
        else:
            return MyAgentAction.MOVE_FORWARD

    def wall_among(things):
        return bool([thing for thing in things if thing.__class__.__name__ == 'Wall'])

    return program
