from assignment.custom_agents.goal_based import GoalBasedAgentProgram, Goal, MyGoalBasedAgent
from assignment.custom_agents.model import ModelBasedAgentProgram, update, Model, MyModalBasedAgent
from assignment.custom_agents.reflex import ReflexAgentProgram, MyReflexAgent
from assignment.environment import MyWorld, Food
from assignment.environment.things import Wall, Guard, Dog, Bark
from assignment.search.agent_using_algo import MyReflexAgentForAlgoProgram
from assignment.search.utils import GraphUtils
from logic import FolKB
from search import UndirectedGraph, GraphProblem, depth_first_graph_search, iterative_deepening_search, \
    breadth_first_graph_search, uniform_cost_search, astar_search, best_first_graph_search, recursive_best_first_search, \
    depth_limited_search
from utils import expr


def test_reflex_agent_program(headless=False):
    program = ReflexAgentProgram()
    agent = MyReflexAgent(program)
    environment = MyWorld(20, 20, color={"Food": [200, 0, 0], "MyReflexAgent": [0, 0, 0], "Wall": [0, 0, 200],
                                         'Marker': [200, 200, 0], "Guard": [0, 200, 0], "Dog": [150, 150, 0],
                                         "Bark": [50, 150, 120]}, headless=headless)
    environment.add_thing(agent, [2, 2])
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

    environment.add_thing(Food(), [18, 18])
    environment.add_thing(Dog(), [16, 15])
    environment.add_thing(Bark(), [17, 16])
    environment.add_thing(Bark(), [15, 16])
    environment.add_thing(Bark(), [16, 16])
    environment.add_thing(Bark(), [17, 15])
    environment.add_thing(Bark(), [15, 15])
    environment.add_thing(Guard(), [7, 18])
    environment.add_thing(Guard(), [18, 15])
    environment.add_thing(Guard(), [14, 18])
    environment.add_walls()
    environment.run(500, 0)
    print("Actions count:", environment.count)


def test_model_based_agent_program(headless=False):
    program = ModelBasedAgentProgram(update, Model())
    agent = MyModalBasedAgent(program)
    environment = MyWorld(20, 20, color={"Food": [200, 0, 0], "MyModalBasedAgent": [0, 0, 0], "Wall": [0, 0, 200],
                                         'Marker': [200, 200, 0], "Guard": [0, 200, 0], "Dog": [150, 150, 0],
                                         "Bark": [50, 150, 120]}, headless=headless)
    environment.add_thing(agent, [2, 2])
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

    environment.add_thing(Food(), [18, 18])
    environment.add_thing(Dog(), [16, 15])
    environment.add_thing(Bark(), [17, 16])
    environment.add_thing(Bark(), [15, 16])
    environment.add_thing(Bark(), [16, 16])
    environment.add_thing(Bark(), [17, 15])
    environment.add_thing(Bark(), [15, 15])
    environment.add_thing(Guard(), [7, 18])
    environment.add_thing(Guard(), [18, 15])
    environment.add_thing(Guard(), [14, 18])
    environment.add_walls()
    environment.run(500, 0)
    print("Actions count:", environment.count)


def test_goal_based_agent_program(headless=False):
    program = GoalBasedAgentProgram(update, Model(), Goal([18, 18]))
    agent = MyGoalBasedAgent(program)
    environment = MyWorld(20, 20, color={"Food": [200, 0, 0], "MyReflexAgent": [0, 0, 0], "Wall": [0, 0, 200],
                                         'Marker': [200, 200, 0], "Guard": [0, 200, 0], "Dog": [255, 255, 255],
                                         "Bark": [255,165,0]}, headless=headless)
    environment.add_thing(agent, [2, 2])
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

    environment.add_thing(Food(), [18, 18])
    environment.add_thing(Dog(), [16, 15])
    environment.add_thing(Bark(), [17, 16])
    environment.add_thing(Bark(), [15, 16])
    environment.add_thing(Bark(), [16, 16])
    environment.add_thing(Bark(), [17, 15])
    environment.add_thing(Bark(), [15, 15])
    environment.add_thing(Guard(), [7, 18])
    environment.add_thing(Guard(), [18, 15])
    environment.add_thing(Guard(), [14, 18])
    environment.add_walls()
    environment.run(500, 0)
    print("Actions count:", environment.count)


def test_graph():
    program = ReflexAgentProgram()
    agent = MyReflexAgent(program)
    environment = MyWorld(20, 20, color={"Food": [200, 0, 0], "MyReflexAgent": [0, 0, 0], "Wall": [0, 0, 200],
                                         'Marker': [200, 200, 0], "Guard": [0, 200, 0],})
    environment.add_thing(agent, [2, 2])
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

    environment.add_thing(Food(), [18, 18])
    environment.add_thing(Dog(), [16, 15])
    environment.add_thing(Bark(), [17, 16])
    environment.add_thing(Bark(), [15, 16])
    environment.add_thing(Bark(), [16, 16])
    environment.add_thing(Bark(), [17, 15])
    environment.add_thing(Bark(), [15, 15])
    environment.add_thing(Guard(), [7, 18])
    environment.add_thing(Guard(), [18, 15])
    environment.add_thing(Guard(), [14, 18])
    environment.add_walls()

    graph_dict = GraphUtils.get_graph_dict(environment.get_world())
    graph = UndirectedGraph(graph_dict)
    problem = GraphProblem(tuple(agent.location), tuple([18, 18]), graph)
    uniform_algos = [depth_first_graph_search, breadth_first_graph_search, depth_limited_search]
    non_uniform_alogs = [recursive_best_first_search, uniform_cost_search, astar_search]
    # import itertools
    # for algo in itertools.chain(uniform_algos, non_uniform_alogs):
    #     print("Algo: ", algo)
    #     sol = algo(problem).solution()
    #     print(sol)
    sol = non_uniform_alogs[0](problem).solution()
    print(sol)
    return sol


def agent_ruuning_in_algo_path(path):
    program = MyReflexAgentForAlgoProgram(path)
    agent = MyReflexAgent(program)
    environment = MyWorld(20, 20, color={"Food": [200, 0, 0], "MyReflexAgent": [0, 0, 0], "Wall": [0, 0, 200],
                                         'Marker': [200, 200, 0], "Guard": [0, 200, 0], })
    environment.add_thing(agent, [2, 2])
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

    environment.add_thing(Food(), [18, 18])
    environment.add_thing(Guard(), [7, 18])
    environment.add_thing(Guard(), [18, 15])
    environment.add_thing(Guard(), [14, 18])
    environment.add_walls()
    environment.run(500, 0)


def test_inference():
    my_world_kb = FolKB()

    my_world_kb.tell(expr("Wall(x) & Agent(y) ==> moveRight(y, x)"))
    my_world_kb.tell(expr("Path(z) & Agent(y) ==> MoveForward(y, z)"))
    my_world_kb.tell(expr("Food(w) & Agent(y) ==> CanEat(y, w)"))

    class Wall:
        name = 'Wall1'

        def expr(self):
            return f'Wall({self.name})'

    class Agent:
        name = 'Agent'

        def expr(self):
            return f'Agent({self.name})'

    class Food:
        name = 'Food1'

        def expr(self):
            return f'Food({self.name})'

    agent = Agent()
    wall = Wall()
    things = [wall, Food()]

    my_world_kb.tell(expr(agent.expr()))
    for thing in things:
        my_world_kb.tell(expr(thing.expr()))

    print(my_world_kb.ask(expr("Avoids(x, y)")))
    print(my_world_kb.ask(expr("MoveForward(Agent, Wall1)")))
    print(my_world_kb.ask(expr("CanEat(Agent, Wall1)")))
    print(my_world_kb.ask(expr("Avoids(Agent, Food1)")))
    print(my_world_kb.ask(expr("MoveForward(Agent, Food1)")))
    print(my_world_kb.ask(expr("CanEat(Agent, Food1)")))


def test_inference2():
    kb0 = FolKB([expr('Farmer(Mac)'), expr('Rabbit(Pete)'), expr('(Rabbit(r) & Farmer(f)) ==> Hates(f, r)')])
    kb0.tell(expr('Rabbit(Flopsie)'))
    kb0.retract(expr('Rabbit(Pete)'))
    print(kb0.ask(expr('Hates(Mac, x)')))
    print(kb0.ask(expr('Wife(Pete, x)')))


if __name__ == '__main__':
    path=test_graph()
    agent_ruuning_in_algo_path(path)
