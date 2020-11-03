import copy
from typing import List

from agents import GraphicEnvironment, Thing, Direction
from assignment.custom_agents import MyAgent, MyAgentAction
from assignment.environment.things import Marker, Food


class Percept:
    def __init__(self, agent: 'MyAgent', things: 'List[Thing]'):
        self.agent = agent
        self.things = things
        self.front_loc, self.back_loc, self.left_loc, self.right_loc = self.__get_locations()
        # todo optimize and do in single loop
        self.things_at_front = [thing for thing in things if tuple(thing.location) == tuple(self.front_loc)]
        self.things_at_back = [thing for thing in things if tuple(thing.location) == tuple(self.back_loc)]
        self.things_at_left = [thing for thing in things if tuple(thing.location) == tuple(self.left_loc)]
        self.things_at_right = [thing for thing in things if tuple(thing.location) == tuple(self.right_loc)]

    def __get_locations(self):
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


class MyWorld(GraphicEnvironment):
    def __init__(self, width=10, height=10, boundary=True, color={}, display=False, headless=False):
        super().__init__(width, height, boundary, color, display)
        self.count = 0
        self.headless = headless

    def percept(self, agent: 'MyAgent') -> 'Percept':
        """return a list of things that are in our agent's location"""
        things = [thing for thing, _ in self.things_near(agent.location, radius=1)]
        return Percept(agent, things)

    def execute_action(self, agent: 'MyAgent', action: 'MyAgentAction'):
        """changes the state of the environment based on what the agent does."""

        if action == MyAgentAction.TURN_RIGHT:
            agent.turn(Direction.R)
            self.count += 1
        elif action == MyAgentAction.TURN_LEFT:
            agent.turn(Direction.L)
            self.count += 1
        elif action == MyAgentAction.MOVE_BACKWARD:
            agent.turn(Direction.L)
            agent.turn(Direction.L)
            self.count += 1
        elif action == MyAgentAction.MOVE_FORWARD:
            things = self.list_things_at(agent.location, tclass=Marker)
            if not things:
                self.add_thing(Marker(), copy.deepcopy(agent.location))
            agent.moveforward()
            self.count += 1
        elif action == MyAgentAction.CAN_GRAB:
            things = [thing for thing, _ in self.things_near(agent.location, radius=1) if
                      thing.__class__.__name__ == 'Food']
            if things and agent.can_grab(things[0]):
                # print('{} grabbed {} at location: {}'.format(str(agent)[1:-1], str(things[0])[1:-1], agent.location))
                self.delete_thing(things[0])
                self.count += 1
        elif action == MyAgentAction.CAN_STAB:
            things = [thing for thing, _ in self.things_near(agent.location, radius=1) if
                      thing.__class__.__name__ == 'Guard']
            if things and agent.can_stab(things[0]):
                # print('{} stabbed {} at location: {}'.format(str(agent)[1:-1], str(things[0])[1:-1], agent.location))
                self.delete_thing(things[0])
                self.count += 1

    def draw_world(self):
        # For simulation we don't want to draw the world
        if self.headless:
            return

        self.grid[:] = (200, 200, 200)
        world = self.get_world()
        for x in range(0, len(world)):
            for y in range(0, len(world[x])):
                things = world[x][y]
                if not len(things):
                    continue
                if len(things) > 1:  # Remove marker if there are more than 1 thing at this location
                    things = list(filter(lambda thing: thing.__class__.__name__ != Marker.__name__, things))
                self.grid[y, x] = self.colors[things[-1].__class__.__name__]

    def is_done(self):
        no_edibles = not any(isinstance(thing, Food) for thing in self.things)
        dead_agents = not any(agent.is_alive() for agent in self.agents)
        return dead_agents or no_edibles
