from abc import ABC
from enum import Enum

from agents import Agent, Direction


class MyAgent(Agent, ABC):
    def __init__(self, program=None):
        super().__init__(program)
        self.location = []
        self.direction = Direction(Direction.D)

    def moveforward(self):
        """moveforward possible only if success (i.e. valid destination location)"""
        if self.direction.direction == Direction.R:
            self.location[0] += 1
        elif self.direction.direction == Direction.L:
            self.location[0] -= 1
        elif self.direction.direction == Direction.D:
            self.location[1] += 1
        elif self.direction.direction == Direction.U:
            self.location[1] -= 1

    def turn(self, turn_direction):
        self.direction = self.direction + turn_direction

    def can_grab(self, thing):
        """returns True upon success or False otherwise"""
        return thing.__class__.__name__ == 'Food'

    def can_stab(self, thing):
        """returns True upon success or False otherwise"""
        return thing.__class__.__name__ == 'Guard'


class MyAgentAction(Enum):
    TURN_RIGHT = "turn_right"
    TURN_LEFT = "turn_left"
    MOVE_FORWARD = "move_forward"
    MOVE_BACKWARD = "move_backward"
    CAN_GRAB = 'can_grab'
    CAN_STAB = 'can_stab'
