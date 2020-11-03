from agents import *
from notebook import psource


class Treasure1(Thing):
    pass


class Treasure2(Thing):
    pass


class Park2D(GraphicEnvironment):
    def percept(self, agent):
        '''return a list of things that are in our agent's location'''
        things = self.list_things_at(agent.location)
        loc = copy.deepcopy(agent.location)  # find out the target location
        # Check if agent is about to bump into a wall
        if agent.direction.direction == Direction.R:
            loc[0] += 1
        elif agent.direction.direction == Direction.L:
            loc[0] -= 1
        elif agent.direction.direction == Direction.D:
            loc[1] += 1
        elif agent.direction.direction == Direction.U:
            loc[1] -= 1
        if not self.is_inbounds(loc):
            things.append(Bump())
        return things

    def execute_action(self, agent, action):
        """changes the state of the environment based on what the agent does."""
        if type(agent).__name__ == 'SecurityGuard':
            if action == 'turnright':
                #             print('{} decided to {} at location: {}'.format(str(agent)[1:-1], action, agent.location))
                agent.turn(Direction.R)
            elif action == 'turnleft':
                #             print('{} decided to {} at location: {}'.format(str(agent)[1:-1], action, agent.location))
                agent.turn(Direction.L)
            elif action == 'moveforward':
                #             print('{} decided to move {}wards at location: {}'.format(str(agent)[1:-1], agent.direction.direction,
                #                                                                       agent.location))
                agent.moveforward()
            elif action == "grab":
                items = self.list_things_at(agent.location, tclass=Treasure1)
                if len(items) != 0:
                    if agent.grab(items[0]):
                        print('{} grabbed {} at location: {}'
                              .format(str(agent)[1:-1], str(items[0])[1:-1], agent.location))
                        self.delete_thing(items[0])
        elif type(agent).__name__ == 'SecurityDog':
            if action == 'turnright':
                #             print('{} decided to {} at location: {}'.format(str(agent)[1:-1], action, agent.location))
                agent.turn(Direction.R)
            elif action == 'turnleft':
                #             print('{} decided to {} at location: {}'.format(str(agent)[1:-1], action, agent.location))
                agent.turn(Direction.L)
            elif action == 'moveforward':
                #             print('{} decided to move {}wards at location: {}'.format(str(agent)[1:-1], agent.direction.direction,
                #                                                                       agent.location))
                agent.moveforward()
            elif action == "bite":
                items = self.list_things_at(agent.location, tclass=Treasure2)
                if len(items) != 0:
                    if agent.bite(items[0]):
                        print('{} bit {} at location: {}'
                              .format(str(agent)[1:-1], str(items[0])[1:-1], agent.location))
                        self.delete_thing(items[0])
        else:
            if action == 'turnright':
                #             print('{} decided to {} at location: {}'.format(str(agent)[1:-1], action, agent.location))
                agent.turn(Direction.R)
            elif action == 'turnleft':
                #             print('{} decided to {} at location: {}'.format(str(agent)[1:-1], action, agent.location))
                agent.turn(Direction.L)
            elif action == 'moveforward':
                #             print('{} decided to move {}wards at location: {}'.format(str(agent)[1:-1], agent.direction.direction,
                #                                                                       agent.location))
                agent.moveforward()

    def is_done(self):
        '''By default, we're done when we can't find a live agent,
        but to prevent killing our cute dog, we will stop before itself - when there is no more food or water'''
        no_edibles = not any(isinstance(thing, Treasure1) or isinstance(thing, Treasure2) for thing in self.things)
        dead_agents = not any(agent.is_alive() for agent in self.agents)
        return dead_agents or no_edibles


# class WumpusEnvironment(XYEnvironment):
#     pit_probability = 0.2  # Probability to spawn a pit in a location. (From Chapter 7.2)
#
#     # Room should be 4x4 grid of rooms. The extra 2 for walls
#
#     def __init__(self, agent_program, width=6, height=6):
#         super().__init__(width, height)
#         self.init_world(agent_program)
#
#     def init_world(self, program):
#         """Spawn items in the world based on probabilities from the book"""
#
#         "WALLS"
#         self.add_walls()
#
#         "PITS"
#         for x in range(self.x_start, self.x_end):
#             for y in range(self.y_start, self.y_end):
#                 if random.random() < self.pit_probability:
#                     self.add_thing(Pit(), (x, y), True)
#                     self.add_thing(Breeze(), (x - 1, y), True)
#                     self.add_thing(Breeze(), (x, y - 1), True)
#                     self.add_thing(Breeze(), (x + 1, y), True)
#                     self.add_thing(Breeze(), (x, y + 1), True)
#
#         "WUMPUS"
#         w_x, w_y = self.random_location_inbounds(exclude=(1, 1))
#         self.add_thing(Wumpus(lambda x: ""), (w_x, w_y), True)
#         self.add_thing(Stench(), (w_x - 1, w_y), True)
#         self.add_thing(Stench(), (w_x + 1, w_y), True)
#         self.add_thing(Stench(), (w_x, w_y - 1), True)
#         self.add_thing(Stench(), (w_x, w_y + 1), True)
#
#         "GOLD"
#         self.add_thing(Gold(), self.random_location_inbounds(exclude=(1, 1)), True)
#
#         "AGENT"
#         self.add_thing(Explorer(program), (1, 1), True)
#
#     def get_world(self, show_walls=True):
#         """Return the items in the world"""
#         result = []
#         x_start, y_start = (0, 0) if show_walls else (1, 1)
#
#         if show_walls:
#             x_end, y_end = self.width, self.height
#         else:
#             x_end, y_end = self.width - 1, self.height - 1
#
#         for x in range(x_start, x_end):
#             row = []
#             for y in range(y_start, y_end):
#                 row.append(self.list_things_at((x, y)))
#             result.append(row)
#         return result
#
#     def percepts_from(self, agent, location, tclass=Thing):
#         """Return percepts from a given location,
#         and replaces some items with percepts from chapter 7."""
#         thing_percepts = {
#             Gold: Glitter(),
#             Wall: Bump(),
#             Wumpus: Stench(),
#             Pit: Breeze()}
#
#         """Agents don't need to get their percepts"""
#         thing_percepts[agent.__class__] = None
#
#         """Gold only glitters in its cell"""
#         if location != agent.location:
#             thing_percepts[Gold] = None
#
#         result = [thing_percepts.get(thing.__class__, thing) for thing in self.things
#                   if thing.location == location and isinstance(thing, tclass)]
#         return result if len(result) else [None]
#
#     def percept(self, agent):
#         """Return things in adjacent (not diagonal) cells of the agent.
#         Result format: [Left, Right, Up, Down, Center / Current location]"""
#         x, y = agent.location
#         result = []
#         result.append(self.percepts_from(agent, (x - 1, y)))
#         result.append(self.percepts_from(agent, (x + 1, y)))
#         result.append(self.percepts_from(agent, (x, y - 1)))
#         result.append(self.percepts_from(agent, (x, y + 1)))
#         result.append(self.percepts_from(agent, (x, y)))
#
#         """The wumpus gives out a loud scream once it's killed."""
#         wumpus = [thing for thing in self.things if isinstance(thing, Wumpus)]
#         if len(wumpus) and not wumpus[0].alive and not wumpus[0].screamed:
#             result[-1].append(Scream())
#             wumpus[0].screamed = True
#
#         return result
#
#     def execute_action(self, agent, action):
#         """Modify the state of the environment based on the agent's actions.
#         Performance score taken directly out of the book."""
#
#         if isinstance(agent, Explorer) and self.in_danger(agent):
#             return
#
#         agent.bump = False
#         if action in ['TurnRight', 'TurnLeft', 'Forward', 'Grab']:
#             super().execute_action(agent, action)
#             agent.performance -= 1
#         elif action == 'Climb':
#             if agent.location == (1, 1):  # Agent can only climb out of (1,1)
#                 agent.performance += 1000 if Gold() in agent.holding else 0
#                 self.delete_thing(agent)
#         elif action == 'Shoot':
#             """The arrow travels straight down the path the agent is facing"""
#             if agent.has_arrow:
#                 arrow_travel = agent.direction.move_forward(agent.location)
#                 while self.is_inbounds(arrow_travel):
#                     wumpus = [thing for thing in self.list_things_at(arrow_travel)
#                               if isinstance(thing, Wumpus)]
#                     if len(wumpus):
#                         wumpus[0].alive = False
#                         break
#                     arrow_travel = agent.direction.move_forward(agent.location)
#                 agent.has_arrow = False
#
#     def in_danger(self, agent):
#         """Check if Explorer is in danger (Pit or Wumpus), if he is, kill him"""
#         for thing in self.list_things_at(agent.location):
#             if isinstance(thing, Pit) or (isinstance(thing, Wumpus) and thing.alive):
#                 agent.alive = False
#                 agent.performance -= 1000
#                 agent.killed_by = thing.__class__.__name__
#                 return True
#         return False
#
#     def is_done(self):
#         """The game is over when the Explorer is killed
#         or if he climbs out of the cave only at (1,1)."""
#         explorer = [agent for agent in self.custom_agents if isinstance(agent, Explorer)]
#         if len(explorer):
#             if explorer[0].alive:
#                 return False
#             else:
#                 print("Death by {} [-1000].".format(explorer[0].killed_by))
#         else:
#             print("Explorer climbed out {}."
#                   .format("with Gold [+1000]!" if Gold() not in self.things else "without Gold [+0]"))
#         return True
#
#     # TODO: Arrow needs to be implemented
#
class Thief(Agent):
    location = [15, 15]
    direction = Direction("down")

    def moveforward(self, success=True):
        '''moveforward possible only if success (i.e. valid destination location)'''
        if not success:
            return
        if self.direction.direction == Direction.R:
            self.location[0] += 1
        elif self.direction.direction == Direction.L:
            self.location[0] -= 1
        elif self.direction.direction == Direction.D:
            self.location[1] += 1
        elif self.direction.direction == Direction.U:
            self.location[1] -= 1

    def turn(self, d):
        self.direction = self.direction + d

    def grab(self, thing):
        """returns True upon success or False otherwise"""
        if isinstance(thing, Treasure1):
            return True
        return False


def program(percepts):
    """Returns an action based on it's percepts"""
    for p in percepts:  # first eat or drink - you're a dog!
        if isinstance(p, Treasure1):
            return 'grab'
        if isinstance(p, Bump):  # then check if you are at an edge and have to turn
            turn = False
            choice_1 = random.choice((1, 2));
        else:
            choice_1 = random.choice((1, 2, 3, 4))  # 1-right, 2-left, others-forward
    if choice_1 == 1:
        return 'turnright'
    elif choice_1 == 2:
        return 'turnleft'
    else:
        return 'moveforward'


#
#
# class Thief_ModelBasedReflexive(Agent):
#     location = [15, 15]
#     direction = Direction("down")
#
#
#     def moveforward(self, success=True):
#         '''moveforward possible only if success (i.e. valid destination location)'''
#         if not success:
#             return
#         if self.direction.direction == Direction.R:
#             self.location[0] += 1
#         elif self.direction.direction == Direction.L:
#             self.location[0] -= 1
#         elif self.direction.direction == Direction.D:
#             self.location[1] += 1
#         elif self.direction.direction == Direction.U:
#             self.location[1] -= 1
#
#     def turn(self, d):
#         self.direction = self.direction + d
#
#     def grab(self, thing):
#         """returns True upon success or False otherwise"""
#         if isinstance(thing, Treasure1):
#             return True
#         return False
#
#
# def program(percepts):
#     """Returns an action based on it's percepts"""
#     for p in percepts:  # first eat or drink - you're a dog!
#         if isinstance(p, Treasure1):
#             return 'grab'
#         if isinstance(p, Bump):  # then check if you are at an edge and have to turn
#             turn = False
#             choice_1 = random.choice((1, 2));
#         else:
#             choice_1 = random.choice((1, 2, 3, 4))  # 1-right, 2-left, others-forward
#     if choice_1 == 1:
#         return 'turnright'
#     elif choice_1 == 2:
#         return 'turnleft'
#     else:
#         return 'moveforward'
#
#
# def test_WumpusEnvironment():
#     def constant_prog(percept):

#         return percept
#
#     # initialize Wumpus Environment
#     w = WumpusEnvironment(constant_prog)
#
#     # check if things are added properly
#     assert len([x for x in w.things if isinstance(x, Wall)]) == 20
#     assert any(map(lambda x: isinstance(x, Gold), w.things))
#     assert any(map(lambda x: isinstance(x, Explorer), w.things))
#     assert not any(map(lambda x: not isinstance(x, Thing), w.things))
#
#     # check that gold and wumpus are not present on (1,1)
#     assert not any(map(lambda x: isinstance(x, Gold) or isinstance(x, WumpusEnvironment), w.list_things_at((1, 1))))
#
#     # check if w.get_world() segments objects correctly
#     assert len(w.get_world()) == 6
#     for row in w.get_world():
#         assert len(row) == 6
#
#     # start the game!
#     agent = [x for x in w.things if isinstance(x, Explorer)][0]
#     gold = [x for x in w.things if isinstance(x, Gold)][0]
#     pit = [x for x in w.things if isinstance(x, Pit)][0]
#
#     assert not w.is_done()
#
#     # check Walls
#     agent.location = (1, 2)
#     percepts = w.percept(agent)
#     assert len(percepts) == 5
#     assert any(map(lambda x: isinstance(x, Bump), percepts[0]))
#
#     # check Gold
#     agent.location = gold.location
#     percepts = w.percept(agent)
#     assert any(map(lambda x: isinstance(x, Glitter), percepts[4]))
#     agent.location = (gold.location[0], gold.location[1] + 1)
#     percepts = w.percept(agent)
#     assert not any(map(lambda x: isinstance(x, Glitter), percepts[4]))
#
#     # check agent death
#     agent.location = pit.location
#     assert w.in_danger(agent)
#     assert not agent.alive
#     assert agent.killed_by == Pit.__name__
#     assert agent.performance == -1000
#
#     assert w.is_done()


park = Park2D(5, 20, color={'Thief': (180, 0, 0), 'Water': (0, 195, 190),
                            'Food': (195, 115, 40)})  # park width is set to 5, and height to 20
thief = Thief(program)
dogfood = Treasure1()
water = Treasure2()
park.add_thing(thief, [0, 1])
park.add_thing(dogfood, [0, 5])
park.add_thing(water, [0, 7])
print("BlindDog starts at (1,1) facing downwards, lets see if he can find any food!")
park.run(20)
