import os, sys, inspect

current_dir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import copy
import random
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
from itertools import chain
from typing import List, Callable, Tuple

from agents import Thing, Obstacle, GraphicEnvironment, Direction, Agent
# ####################### THINGS IN THE ENVIRONMENT (OTHER THAN PLAYER AGENT) #######################
from logic import FolKB, fol_fc_ask, fol_bc_ask
from search import UndirectedGraph, GraphProblem, depth_first_graph_search, breadth_first_graph_search, \
    depth_limited_search, recursive_best_first_search, uniform_cost_search, astar_search, expr
from utils import first


class Symbolic(ABC):
    """Abstract class that marks a class to be used in First order logic"""

    @abstractmethod
    def expr(self):
        pass


class Treasure(Thing, Symbolic):
    """Goal of the thief agent"""

    def __init__(self):
        self.name = f'{Treasure.__name__}{str(uuid.uuid4())[:5]}'

    def expr(self):
        return expr(f'{Treasure.__name__}({self.name})')


class Marker(Thing, Symbolic):
    """Mark the path traversed by the thief agent"""

    def __init__(self):
        self.name = f'{Marker.__name__}{str(uuid.uuid4())[:5]}'

    def expr(self):
        return expr(f'{Marker.__name__}({self.name})')


class Wall(Obstacle, Symbolic):
    """Prevent any agent from moving into a cell"""

    def __init__(self):
        self.name = f'{Wall.__name__}{str(uuid.uuid4())[:5]}'

    def expr(self):
        return expr(f'{Wall.__name__}({self.name})')


class Guard(Thing, Symbolic):
    """Protects the treasure and will be avoided by thief agent"""

    def __init__(self):
        self.name = f'{Guard.__name__}{str(uuid.uuid4())[:5]}'

    def expr(self):
        return expr(f'{Guard.__name__}({self.name})')


class Dog(Obstacle, Symbolic):
    """Protect sthe treasure and will be avoided by thief agent"""

    def __init__(self):
        self.name = f'{Dog.__name__}{str(uuid.uuid4())[:5]}'

    def expr(self):
        return expr(f'{Dog.__name__}({self.name})')


class Bark(Obstacle, Symbolic):
    """Positioned near a dog and perceived by the thief agent to avoid the dog"""

    def __init__(self):
        self.name = f'{Bark.__name__}{str(uuid.uuid4())[:5]}'

    def expr(self):
        return expr(f'{Bark.__name__}({self.name})')


# ####################### ENVIRONMENT #######################
class TreasureVault(GraphicEnvironment):
    """
    An Environment which contains a treasure, guard(s), dog(s), wall(s) and a thief. Walls will limit access of the
    thief to certain places. Thief will try to loot the treasure while dog(s) and guard(s) will try to protect it.
    """

    def __init__(self, width=10, height=10, boundary=True, color=None, display=False, headless=False):
        super().__init__(width, height, boundary, color or {}, display)
        # Count of actions performed by all agents in this environment. Currently only thief agent is able to
        # execute any action so effectively it contains the count of actions performed by thief agent
        self.agent_actions = 0
        # Toggles the display of GUI. We want `headless` to be `False` during stress/performance testing of our
        # environment. We don't want to be bottle-necked by GUI rendering
        self.headless = headless

    def percept(self, agent: 'TreasureVaultAgent') -> 'TreasureVaultAgentPercept':
        """Returns an object representing information perceived by an agent at its current location"""
        # `things_near` method will return all the things around a given location and radius.
        things = [thing for thing, _ in self.things_near(agent.location, radius=1)]
        return TreasureVaultAgentPercept(agent, things)

    def execute_action(self, agent: 'TreasureVaultAgent', action: 'TreasureVaultAgentAction'):
        """Changes the state of the environment based on what the agent does."""
        if action == TreasureVaultAgentAction.TURN_RIGHT:
            agent.turn(Direction.R)
            self.agent_actions += 1
        elif action == TreasureVaultAgentAction.TURN_LEFT:
            agent.turn(Direction.L)
            self.agent_actions += 1
        elif action == TreasureVaultAgentAction.MOVE_BACKWARD:
            agent.turn(Direction.L)
            agent.turn(Direction.L)
            self.agent_actions += 1
        elif action == TreasureVaultAgentAction.MOVE_FORWARD:
            # Add a `Marker` to the current location of the agent to show the path traversed by it
            things = self.list_things_at(agent.location, tclass=Marker)
            if not things:
                self.add_thing(Marker(), copy.deepcopy(agent.location))
            agent.moveforward()
            self.agent_actions += 1
        elif action == TreasureVaultAgentAction.CAN_GRAB:
            # Get all the treasures at the agents location
            things = [thing for thing, _ in self.things_near(agent.location, radius=1) if
                      thing.__class__.__name__ == Treasure.__name__]
            # If an agent can grab that treasure remove it from the environment
            if things and agent.can_grab(things[0]):
                self.delete_thing(things[0])
                self.agent_actions += 1
        elif action == TreasureVaultAgentAction.CAN_STAB:
            # Get all the guards at the agents location
            things = [thing for thing, _ in self.things_near(agent.location, radius=1) if
                      thing.__class__.__name__ == Guard.__name__]
            # If an agent can grab that treasure remove it from the environment
            if things and agent.can_stab(things[0]):
                self.delete_thing(things[0])
                self.agent_actions += 1

    def draw_world(self):
        """
        Draw the environment
        NOTE: We have overloaded the draw_world method from the parent environment because we wanted our custom
        implementation related to markers left by agents.
        """
        # Used for simulation without actual drawing
        if self.headless:
            return

        self.grid[:] = (200, 200, 200)
        world = self.get_world()
        for x in range(0, len(world)):
            for y in range(0, len(world[x])):
                things = world[x][y]
                if not len(things):
                    continue
                # Remove marker if there are more than 1 thing at this location
                # We don't want our `Marker` to hide any of the agents.
                if len(things) > 1:
                    things = [thing for thing in things if thing.__class__.__name__ != Marker.__name__]
                self.grid[y, x] = self.colors[things[-1].__class__.__name__]

    def add_walls(self):
        """
        Put walls around the entire perimeter of the grid.
        NOTE: We have overrided this method because we want our symbolic walls to be added
        """
        for x in range(self.width):
            self.add_thing(Wall(), (x, 0))
            self.add_thing(Wall(), (x, self.height - 1))
        for y in range(1, self.height - 1):
            self.add_thing(Wall(), (0, y))
            self.add_thing(Wall(), (self.width - 1, y))

        # Updates iteration start and end (with walls).
        self.x_start, self.y_start = (1, 1)
        self.x_end, self.y_end = (self.width - 1, self.height - 1)

    def is_done(self):
        """Returns True if no actions are supposed to be performed on the environment"""
        no_treasures = not any(isinstance(thing, Treasure) for thing in self.things)
        dead_agents = not any(agent.is_alive() for agent in self.agents)
        return dead_agents or no_treasures


# ####################### PLAYER AGENT #######################
class TreasureVaultAgentAction(Enum):
    """Represents the actions which can be performed in `TreasureVault` environment"""
    TURN_RIGHT = "turn_right"
    TURN_LEFT = "turn_left"
    MOVE_FORWARD = "move_forward"
    MOVE_BACKWARD = "move_backward"
    CAN_GRAB = 'can_grab'
    CAN_STAB = 'can_stab'


class TreasureVaultAgentPercept:
    """Represents a percept received by an agent in TreasureVault environment"""

    # noinspection PyUnresolvedReferences
    def __init__(self, agent: 'TreasureVaultAgent', things: 'List[Thing]'):
        self.agent = agent
        self.things = things
        # Get locations which are in-front, back, left, and right of the perceiving agent
        self.front_loc, self.back_loc, self.left_loc, self.right_loc = self.__get_locations()
        # Segregate things which are perceived by an agent on the basis of their direction relative to it
        self.things_at_front = [thing for thing in things if tuple(thing.location) == tuple(self.front_loc)]
        self.things_at_back = [thing for thing in things if tuple(thing.location) == tuple(self.back_loc)]
        self.things_at_left = [thing for thing in things if tuple(thing.location) == tuple(self.left_loc)]
        self.things_at_right = [thing for thing in things if tuple(thing.location) == tuple(self.right_loc)]

    def __get_locations(self):
        """Returns a tuple representing front, back, left and right position of an the perceiving agent"""
        front_loc = copy.deepcopy(self.agent.location)
        back_loc = copy.deepcopy(self.agent.location)
        left_loc = copy.deepcopy(self.agent.location)
        right_loc = copy.deepcopy(self.agent.location)

        if self.agent.direction.direction == Direction.R:
            front_loc[0] += 1
            back_loc[0] -= 1
            left_loc[1] -= 1
            right_loc[1] += 1
        elif self.agent.direction.direction == Direction.L:
            front_loc[0] -= 1
            back_loc[0] += 1
            left_loc[1] += 1
            right_loc[1] -= 1
        elif self.agent.direction.direction == Direction.D:
            front_loc[1] += 1
            back_loc[1] -= 1
            left_loc[0] += 1
            right_loc[0] -= 1
        elif self.agent.direction.direction == Direction.U:
            front_loc[1] -= 1
            back_loc[1] += 1
            left_loc[0] -= 1
            right_loc[0] += 1

        return front_loc, back_loc, left_loc, right_loc


class TreasureVaultAgent(Agent):
    """Base class for agents in the 'TreasureVault' environment"""

    def __init__(self, program=None):
        super().__init__(program)
        self.location = []
        # By default it will be facing downwards
        self.direction = Direction(Direction.D)

    def moveforward(self):
        """Agent will move forward according to the current direction"""
        if self.direction.direction == Direction.R:
            self.location[0] += 1
        elif self.direction.direction == Direction.L:
            self.location[0] -= 1
        elif self.direction.direction == Direction.D:
            self.location[1] += 1
        elif self.direction.direction == Direction.U:
            self.location[1] -= 1

    def turn(self, turn_direction):
        """Agent will move forward according to the `turn_direction`"""
        self.direction = self.direction + turn_direction

    def can_grab(self, thing):
        """Returns True if agent can grab the input `thing`"""
        return thing.__class__.__name__ == Treasure.__name__

    def can_stab(self, thing):
        """Returns True if agent can stab the input `thing`"""
        return thing.__class__.__name__ == Guard.__name__


class Thief(TreasureVaultAgent):
    """Will try to steal the treasure in TreasureVaultEnvironment while avoiding walls, guards and dogs"""
    pass


# ####################### SIMPLE REFLEX AGENT PROGRAM #######################

# noinspection PyPep8Naming
def ReflexAgentProgram():
    def program(percept: 'TreasureVaultAgentPercept'):
        return rule_match(percept)

    def rule_match(percept: 'TreasureVaultAgentPercept') -> 'TreasureVaultAgentAction':
        """Returns an action on the basis of percept"""
        for thing in percept.things:
            if isinstance(thing, Treasure):
                return TreasureVaultAgentAction.CAN_GRAB
            elif isinstance(thing, Guard):
                return TreasureVaultAgentAction.CAN_STAB

        if to_avoid(percept.things_at_front):
            # Probabilities of actions: TURN_RIGHT(50%), TURN_LEFT(50%)
            choice = random.choice([1, 2])
        else:
            # Probabilities of actions: TURN_RIGHT(25%), TURN_LEFT(25%), MOVE_FORWARD(50%)
            choice = random.choice((1, 2, 3, 4))

        if choice == 1:
            return TreasureVaultAgentAction.TURN_RIGHT
        elif choice == 2:
            return TreasureVaultAgentAction.TURN_LEFT
        else:
            return TreasureVaultAgentAction.MOVE_FORWARD

    def to_avoid(things):
        """Returns True if there is no walls among the give things"""
        avoided_things = {Wall.__name__, Dog.__name__, Bark.__name__}
        return bool([thing for thing in things if thing.__class__.__name__ in avoided_things])

    return program


# ####################### MODEL-BASED AGENT PROGRAM #######################

class TreasureVaultAgentModal:
    """Represents agent's model"""

    def __init__(self):
        self.history = defaultdict(list)  # Mapping of the location and action taken by an agent
        self.visited = set()  # Set of locations visited by an agent


def update(last_percept: 'TreasureVaultAgentPercept', last_action: 'TreasureVaultAgentAction',
           percept: 'TreasureVaultAgentPercept', model: 'TreasureVaultAgentModal'):
    """Helper method to update the agent's model according to last percpet and actions, and current percept"""
    if not last_percept:
        return

    # Update model
    model.history[tuple(last_percept.agent.location)].append(last_action)
    # If the last action was MOVE_FORWARD then it implies we have to mark current location as visited because agent
    # must have moved to it in last step
    if last_action == TreasureVaultAgentAction.MOVE_FORWARD:
        model.visited.add(tuple(percept.agent.location))


# noinspection PyPep8Naming
def ModelBasedAgentProgram(update_state: 'Callable', model: 'TreasureVaultAgentModal'):
    def program(percept: 'TreasureVaultAgentPercept'):
        # Update the model according to last percept and action, and current percept and store the current percept
        update_state(program.last_percept, program.last_action, percept, model)
        # Get action from the current percept and model
        action = rule_match(percept, model)
        # Save the current percept and action
        program.last_percept = percept
        program.last_action = action

        return action

    program.last_percept = program.last_action = None

    def rule_match(percept: 'TreasureVaultAgentPercept',
                   model: 'TreasureVaultAgentModal') -> 'TreasureVaultAgentAction':
        """Returns an action on the basis of percept and model"""
        for thing in percept.things:
            if isinstance(thing, TreasureVaultAgentAction):
                return TreasureVaultAgentAction.CAN_GRAB
            elif isinstance(thing, Guard):
                return TreasureVaultAgentAction.CAN_STAB

        # Turn if there is a thing to avoid in front of the agent or the front location is already visited
        if tuple(percept.front_loc) in model.visited or to_avoid(percept.things_at_front):
            # Probabilities of actions: TURN_RIGHT(50%), TURN_LEFT(50%)
            choice = random.choice((1, 2))
        else:
            # Probabilities of actions: TURN_RIGHT(25%), TURN_LEFT(25%), MOVE_FORWARD(50%)
            choice = random.choice((1, 2, 3, 4))

        if choice == 1:
            return TreasureVaultAgentAction.TURN_RIGHT
        elif choice == 2:
            return TreasureVaultAgentAction.TURN_LEFT
        else:
            return TreasureVaultAgentAction.MOVE_FORWARD

    def to_avoid(things):
        """Returns True if there is walls among the give things"""
        avoided_things = {Wall.__name__, Dog.__name__, Bark.__name__}
        return bool([thing for thing in things if thing.__class__.__name__ in avoided_things])

    return program


# ####################### GOAL-BASED AGENT PROGRAM #######################

class ThiefGoal:
    """Goal of the thief """

    def __init__(self, goal_loc: 'List'):
        self.goal_loc = goal_loc

    def distance(self, location: 'List') -> int:
        """Manhattan distance from the given location to the goal location"""
        return abs(location[0] - self.goal_loc[0]) + abs(location[1] - self.goal_loc[1])


# noinspection PyPep8Naming
def GoalBasedAgentProgram(update_state: 'Callable', model: 'TreasureVaultAgentModal', goal: 'ThiefGoal'):
    def program(percept: 'TreasureVaultAgentPercept'):
        # Update the model according to last percept and action, and current percept and store the current percept
        update_state(program.last_percept, program.last_action, percept, model)
        # Get action from the current percept and model
        action = rule_match(percept, model, goal)
        # Save the current percept and action
        program.last_percept = percept
        program.last_action = action
        return action

    program.last_percept = program.last_action = None

    def rule_match(percept: 'TreasureVaultAgentPercept', model: 'TreasureVaultAgentModal',
                   goal: 'ThiefGoal') -> 'TreasureVaultAgentAction':
        """Returns an action on the basis of percept, model and goal"""
        for thing in percept.things:
            if isinstance(thing, Treasure):
                return TreasureVaultAgentAction.CAN_GRAB
            elif isinstance(thing, Guard):
                return TreasureVaultAgentAction.CAN_STAB

        # It will use a greedy search powered by Manhattan distance to decide it's next action according to the goal
        # It will check the distance from the three blocks (front, left, and right) to the goal-state and decide the
        # one with lowest distance.
        # In case of equal distance the MOVE_FORWARD action takes precedence. For example if distance from right cell
        # and front cells are say 5 then agent will MOVE_FORWARD instead of TURN_RIGHT
        decision_dict = dict()  # Mapping of distances and corresponding actions
        if no_things_to_avoid(percept.things_at_front):
            decision_dict[goal.distance(percept.front_loc)] = TreasureVaultAgentAction.MOVE_FORWARD
        if no_things_to_avoid(percept.things_at_right):
            distance = goal.distance(percept.right_loc)
            # Ensure that MOVE_FORWARD has the precedence
            if distance not in decision_dict:
                decision_dict[goal.distance(percept.right_loc)] = TreasureVaultAgentAction.TURN_RIGHT
        if no_things_to_avoid(percept.things_at_left):
            distance = goal.distance(percept.left_loc)
            # Ensure that MOVE_FORWARD has the precedence
            if distance not in decision_dict:
                decision_dict[goal.distance(percept.left_loc)] = TreasureVaultAgentAction.TURN_LEFT

        # Sort the distances in ascending order
        distances = sorted(list(decision_dict.keys()))
        # If there are no distances it means at all the three cells: front, left and right there are obstacles. So in
        # this scenario jut turn right and eventually agent can move forward
        if not distances:
            return TreasureVaultAgentAction.TURN_RIGHT

        # Get the action corresponding to the lowest distance
        return decision_dict[distances[0]]

    def no_things_to_avoid(things):
        """Returns True if there is no walls among the give things"""
        avoided_things = {Wall.__name__, Dog.__name__, Bark.__name__}
        return not bool([thing for thing in things if thing.__class__.__name__ in avoided_things])

    return program


# ####################### SEARCH-BASED AGENT PROGRAM #######################

class GraphUtils:
    """Utility class to create graph from our problem"""

    @staticmethod
    def get_graph_dict(world_matrix: 'List[List[List[Thing]]]') -> dict:
        """Get a graph representation  from matrix representation of an environment"""
        graph_dict = dict()
        row, col = len(world_matrix), len(world_matrix[0])

        for x in range(row):
            for y in range(col):
                node = x, y
                graph_dict[node] = dict()

                if GraphUtils.__to_avoid(world_matrix[node[0]][node[1]]):
                    continue

                neighbours = GraphUtils.__get_neighbours(*node)  # Get neighbours for current location
                # Filter invalid neighbours
                neighbours = [neighbour for neighbour in neighbours if
                              GraphUtils.__isvalid_coordinates(neighbour[0], neighbour[1], row, col)]
                # Filter neighbours which contain walls
                neighbours = [neighbour for neighbour in neighbours if
                              not GraphUtils.__to_avoid(world_matrix[neighbour[0]][neighbour[1]])]
                # Add the edge for current node and its neighbour in the graph.
                # We are assuming constant path cost of 1
                for neighbour in neighbours:
                    graph_dict[node][neighbour] = 1

        return graph_dict

    @staticmethod
    def __get_neighbours(x: int, y: int):
        """Get neighbours for a x, y coordinate"""
        return (x + 1, y), (x, y + 1), (x - 1, y), (x, y - 1)

    @staticmethod
    def __isvalid_coordinates(x: int, y: int, max_x: int, max_y: int) -> bool:
        """Returns True if the given x, y coordinate is outside our world"""
        return 0 <= x < max_x and 0 <= y < max_y

    @staticmethod
    def __to_avoid(things):
        """Returns True if there is walls among the give things"""
        avoided_things = {Wall.__name__, Dog.__name__, Bark.__name__}
        return bool([thing for thing in things if thing.__class__.__name__ in avoided_things])


# noinspection PyPep8Naming
def SearchBasedAgentProgram(path_to_follow: 'List[Tuple]'):
    def program(percept: 'TreasureVaultAgentPercept'):
        return rule_match(percept)

    def rule_match(percept: 'TreasureVaultAgentPercept') -> 'TreasureVaultAgentAction':
        """Returns an action based on it's percepts"""

        for thing in percept.things:
            if isinstance(thing, Treasure):
                return TreasureVaultAgentAction.CAN_GRAB
            elif isinstance(thing, Guard):
                return TreasureVaultAgentAction.CAN_STAB

        # Possible locations for the agent
        possible_locations = [tuple(percept.front_loc), tuple(percept.back_loc), tuple(percept.right_loc),
                              tuple(percept.left_loc)]

        point = None

        for possible_location in possible_locations:
            if possible_location in path_to_follow:
                point = possible_location
                break

        # Point is never None because our environment is static, so we will always find a point to move.
        # In case of dynamic environments we have to take a search algorithm as an input to the program instead of a
        # path as the point has to be calculated at every step
        assert point

        if percept.right_loc == list(point):
            return TreasureVaultAgentAction.TURN_RIGHT
        elif percept.left_loc == list(point):
            return TreasureVaultAgentAction.TURN_LEFT
        elif percept.front_loc == list(point):
            path_to_follow.remove(point)  # After moving forward we will remove it from the `path_to_follow`
            return TreasureVaultAgentAction.MOVE_FORWARD
        elif percept.back_loc == list(point):
            return TreasureVaultAgentAction.MOVE_BACKWARD

    return program


# noinspection PyPep8Naming
def InferenceBasedAgentProgram(ask_fn: 'Callable', kb: 'FolKB', agent_name: str):
    def program(percept: 'TreasureVaultAgentPercept'):
        return rule_match(percept, ask_fn, kb, agent_name)

    def rule_match(percept: 'TreasureVaultAgentPercept', ask_fn: 'Callable', kb: 'FolKB',
                   agent_name: str) -> 'TreasureVaultAgentAction':
        """Returns an action on the basis of percept"""
        for thing in percept.things:
            if not isinstance(thing, Symbolic):
                continue

            kb.tell(thing.expr())  # Add the symbol for current thing to kb
            # Ask for inferences from kb
            if __is_condition_satisfied(ask_fn, kb, agent_name, thing, 'CanGrab'):
                return TreasureVaultAgentAction.CAN_GRAB
            if __is_condition_satisfied(ask_fn, kb, agent_name, thing, 'CanStab'):
                return TreasureVaultAgentAction.CAN_STAB

        to_avoid = False
        for thing in percept.things_at_front:
            if not isinstance(thing, Symbolic):
                continue

            kb.tell(thing.expr())  # Add the symbol for current thing to kb
            # Ask for inferences from kb
            if __is_condition_satisfied(ask_fn, kb, agent_name, thing, 'Avoids'):
                to_avoid = True
                break

        if to_avoid:
            # Probabilities of actions: TURN_RIGHT(50%), TURN_LEFT(50%)
            choice = random.choice([1, 2])
        else:
            # Probabilities of actions: TURN_RIGHT(25%), TURN_LEFT(25%), MOVE_FORWARD(50%)
            choice = random.choice((1, 2, 3, 4))

        if choice == 1:
            return TreasureVaultAgentAction.TURN_RIGHT
        elif choice == 2:
            return TreasureVaultAgentAction.TURN_LEFT
        else:
            return TreasureVaultAgentAction.MOVE_FORWARD

    def __is_condition_satisfied(ask_fn: 'Callable', kb: 'FolKB', agent_name: 'str',
                                 symbolic_thing, condition: 'str') -> bool:
        """
        Check whether a condition expression of syntax: Condition(Thief(x), <Symbolic>(y)) satisfies
        For ex:

        To check for 'Avoids' for a Symbolic with name "Wall123" we get an answer using the `ask_fn`
        answer = ask_fn("Avoids(x, Wall123)")
        so if in the answer dict we can get a unification of "x" with the name of agent then it means
        that the agent satisfies the condition with that symbolic
        """
        answer = first(ask_fn(kb, expr(f"{condition}(x, {symbolic_thing.name})")), default={})
        return answer.get(expr('x')) == expr(agent_name)

    return program


# ####################### QUESTION 1 #######################
def run_simple_reflex_agent_program(runs=500, delay=0.1, headless=False):
    program = ReflexAgentProgram()
    agent = Thief(program)
    environment = TreasureVault(20, 20,
                                color={Treasure.__name__: [200, 0, 0], Thief.__name__: [0, 0, 0],
                                       Wall.__name__: [0, 0, 200],
                                       Marker.__name__: [200, 200, 0], Guard.__name__: [0, 200, 0],
                                       Dog.__name__: [150, 150, 0],
                                       Bark.__name__: [50, 150, 120]}, headless=headless)

    # Add agent
    environment.add_thing(agent, [2, 2])

    # Add walls
    environment.add_walls()
    for i in range(1, 8):
        environment.add_thing(Wall(), [i, 4])
        environment.add_thing(Wall(), [i, 5])
    for i in range(10, 18):
        environment.add_thing(Wall(), [5, i])
        environment.add_thing(Wall(), [6, i])
    for i in range(2, 10):
        environment.add_thing(Wall(), [14, i])
        environment.add_thing(Wall(), [15, i])
    for i in range(11, 18):
        environment.add_thing(Wall(), [i, 13])
        environment.add_thing(Wall(), [i, 14])

    # Add treasure
    environment.add_thing(Treasure(), [18, 18])

    # Add dogs and guards
    environment.add_thing(Dog(), [16, 15])
    environment.add_thing(Bark(), [17, 16])
    environment.add_thing(Bark(), [15, 16])
    environment.add_thing(Bark(), [16, 16])
    environment.add_thing(Bark(), [17, 15])
    environment.add_thing(Bark(), [15, 15])
    environment.add_thing(Guard(), [7, 18])
    environment.add_thing(Guard(), [18, 15])
    environment.add_thing(Guard(), [14, 18])

    environment.run(runs, delay)
    print("Actions count:", environment.agent_actions)


def run_model_based_agent_program(runs=500, delay=0.1, headless=False):
    program = ModelBasedAgentProgram(update, TreasureVaultAgentModal())
    agent = Thief(program)
    environment = TreasureVault(20, 20, color={Treasure.__name__: [200, 0, 0], Thief.__name__: [0, 0, 0],
                                               Wall.__name__: [0, 0, 200],
                                               Marker.__name__: [200, 200, 0], Guard.__name__: [0, 200, 0],
                                               Dog.__name__: [150, 150, 0],
                                               Bark.__name__: [50, 150, 120]}, headless=headless)

    # Add agent
    environment.add_thing(agent, [2, 2])

    # Add walls
    environment.add_walls()
    for i in range(1, 8):
        environment.add_thing(Wall(), [i, 4])
        environment.add_thing(Wall(), [i, 5])
    for i in range(10, 18):
        environment.add_thing(Wall(), [5, i])
        environment.add_thing(Wall(), [6, i])
    for i in range(2, 10):
        environment.add_thing(Wall(), [14, i])
        environment.add_thing(Wall(), [15, i])
    for i in range(11, 18):
        environment.add_thing(Wall(), [i, 13])
        environment.add_thing(Wall(), [i, 14])

    # Add treasure
    environment.add_thing(Treasure(), [18, 18])

    # Add dogs and guards
    environment.add_thing(Dog(), [16, 15])
    environment.add_thing(Bark(), [17, 16])
    environment.add_thing(Bark(), [15, 16])
    environment.add_thing(Bark(), [16, 16])
    environment.add_thing(Bark(), [17, 15])
    environment.add_thing(Bark(), [15, 15])
    environment.add_thing(Guard(), [7, 18])
    environment.add_thing(Guard(), [18, 15])
    environment.add_thing(Guard(), [14, 18])

    environment.run(runs, delay)
    print("Actions count:", environment.agent_actions)


def run_goal_based_agent_program(runs=500, delay=0.1, headless=False):
    treasure_location = [18, 18]
    program = GoalBasedAgentProgram(update, TreasureVaultAgentModal(), ThiefGoal(copy.deepcopy(treasure_location)))
    agent = Thief(program)
    environment = TreasureVault(20, 20, color={Treasure.__name__: [200, 0, 0], Thief.__name__: [0, 0, 0],
                                               Wall.__name__: [0, 0, 200],
                                               Marker.__name__: [200, 200, 0], Guard.__name__: [0, 200, 0],
                                               Dog.__name__: [255, 255, 255],
                                               Bark.__name__: [255, 165, 0]}, headless=headless)
    # Add agent
    environment.add_thing(agent, [2, 2])

    # Add walls
    environment.add_walls()
    for i in range(1, 8):
        environment.add_thing(Wall(), [i, 4])
        environment.add_thing(Wall(), [i, 5])
    for i in range(10, 18):
        environment.add_thing(Wall(), [5, i])
        environment.add_thing(Wall(), [6, i])
    for i in range(2, 10):
        environment.add_thing(Wall(), [14, i])
        environment.add_thing(Wall(), [15, i])
    for i in range(11, 18):
        environment.add_thing(Wall(), [i, 13])
        environment.add_thing(Wall(), [i, 14])

    # Add treasure
    environment.add_thing(Treasure(), copy.deepcopy(treasure_location))

    # Add dogs and guards
    environment.add_thing(Dog(), [16, 15])
    environment.add_thing(Bark(), [17, 16])
    environment.add_thing(Bark(), [15, 16])
    environment.add_thing(Bark(), [16, 16])
    environment.add_thing(Bark(), [17, 15])
    environment.add_thing(Bark(), [15, 15])
    environment.add_thing(Guard(), [7, 18])
    environment.add_thing(Guard(), [18, 15])
    environment.add_thing(Guard(), [14, 18])

    environment.run(runs, delay)
    print("Actions count:", environment.agent_actions)


def q1():
    run_simple_reflex_agent_program()
    run_model_based_agent_program()
    run_goal_based_agent_program()


# ####################### QUESTION 2 #######################

def run_search_based_agent_program(search_algo, runs=500, delay=0.1, headless=False):
    agent_location = [2, 2]
    treasure_location = [18, 18]
    environment = TreasureVault(20, 20, color={Treasure.__name__: [200, 0, 0], Thief.__name__: [0, 0, 0],
                                               Wall.__name__: [0, 0, 200],
                                               Marker.__name__: [200, 200, 0], Guard.__name__: [0, 200, 0]},
                                headless=headless)

    # Adding walls
    environment.add_walls()
    for i in range(1, 8):
        environment.add_thing(Wall(), [i, 4])
        environment.add_thing(Wall(), [i, 5])
    for i in range(10, 18):
        environment.add_thing(Wall(), [5, i])
        environment.add_thing(Wall(), [6, i])
    for i in range(2, 10):
        environment.add_thing(Wall(), [14, i])
        environment.add_thing(Wall(), [15, i])
    for i in range(11, 18):
        environment.add_thing(Wall(), [i, 13])
        environment.add_thing(Wall(), [i, 14])

    # Adding treasure
    environment.add_thing(Treasure(), copy.deepcopy(treasure_location))

    # Adding guards
    environment.add_thing(Guard(), [7, 18])
    environment.add_thing(Guard(), [18, 15])
    environment.add_thing(Guard(), [14, 18])

    # Creating graph from our environment
    graph_dict = GraphUtils.get_graph_dict(environment.get_world())
    graph = UndirectedGraph(graph_dict)

    # Getting the search path from the given search algorithm in our graph between the agent and the treasure location
    problem = GraphProblem(tuple(agent_location), tuple(treasure_location), graph)
    path_to_follow = search_algo(problem).solution()

    # Adding the agent
    program = SearchBasedAgentProgram(path_to_follow)
    agent = Thief(program)
    environment.add_thing(agent, agent_location)

    environment.run(runs, delay)


def q2():
    uniform_algos = [depth_first_graph_search, breadth_first_graph_search, depth_limited_search]
    non_uniform_alogs = [recursive_best_first_search, uniform_cost_search, astar_search]

    for algo in chain(uniform_algos, non_uniform_alogs):
        print("Running algo: ", algo.__name__)
        run_search_based_agent_program(algo)


# ####################### QUESTION 3 #######################
def run_inference_based_agent(ask_fn, runs=500, delay=0.1, headless=False):
    treasure_vault_kb = FolKB()
    treasure_vault_kb.tell(expr(f"Bark(x) & Thief(y) ==> Avoids(y, x)"))
    treasure_vault_kb.tell(expr(f"Dog(x) & Thief(y) ==> Avoids(y, x)"))
    treasure_vault_kb.tell(expr(f"Wall(x) & Thief(y) ==> Avoids(y, x)"))
    treasure_vault_kb.tell(expr(f"Guard(x) & Thief(y) ==> CanStab(y, x)"))
    treasure_vault_kb.tell(expr(f"Treasure(x) & Thief(y) ==> CanGrab(y, x)"))

    agent_name = f'{Thief.__name__}{str(uuid.uuid4())[:5]}'
    agent_expr = expr(f'{Thief.__name__}({agent_name})')
    treasure_vault_kb.tell(agent_expr)

    program = InferenceBasedAgentProgram(ask_fn, treasure_vault_kb, agent_name)
    agent = Thief(program)
    environment = TreasureVault(20, 20,
                                color={Treasure.__name__: [200, 0, 0], Thief.__name__: [0, 0, 0],
                                       Wall.__name__: [0, 0, 200],
                                       Marker.__name__: [200, 200, 0], Guard.__name__: [0, 200, 0],
                                       Dog.__name__: [150, 150, 0],
                                       Bark.__name__: [50, 150, 120]}, headless=headless)

    # Add agent
    environment.add_thing(agent, [2, 2])

    # Add walls
    environment.add_walls()
    for i in range(1, 8):
        environment.add_thing(Wall(), [i, 4])
        environment.add_thing(Wall(), [i, 5])
    for i in range(10, 18):
        environment.add_thing(Wall(), [5, i])
        environment.add_thing(Wall(), [6, i])
    for i in range(2, 10):
        environment.add_thing(Wall(), [14, i])
        environment.add_thing(Wall(), [15, i])
    for i in range(11, 18):
        environment.add_thing(Wall(), [i, 13])
        environment.add_thing(Wall(), [i, 14])

    # Add treasure
    environment.add_thing(Treasure(), [18, 18])

    # Add dogs and guards
    environment.add_thing(Dog(), [16, 15])
    environment.add_thing(Bark(), [17, 16])
    environment.add_thing(Bark(), [15, 16])
    environment.add_thing(Bark(), [16, 16])
    environment.add_thing(Bark(), [17, 15])
    environment.add_thing(Bark(), [15, 15])
    environment.add_thing(Guard(), [7, 18])
    environment.add_thing(Guard(), [18, 15])
    environment.add_thing(Guard(), [14, 18])

    environment.run(runs, delay)
    print("Actions count:", environment.agent_actions)


def q3():
    ask_fns = [fol_bc_ask, fol_fc_ask]
    for ask_fn in ask_fns:
        run_inference_based_agent(ask_fn)


if __name__ == '__main__':
    # NOTE: Uncomment the corresponding question to run
    # q1()
    # q2()
    q3()
    pass