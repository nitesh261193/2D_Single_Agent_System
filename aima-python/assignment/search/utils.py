from typing import List

from agents import Thing


class GraphUtils:

    @staticmethod
    def get_graph_dict(world_matrix: 'List[List[List[Thing]]]') -> dict:
        graph_dict = dict()
        row, col = len(world_matrix), len(world_matrix[0])
        for x in range(row):
            for y in range(col):
                node = x, y
                graph_dict[node] = dict()
                if GraphUtils.__wall_among(world_matrix[node[0]][node[1]]):
                    continue
                neighbours = GraphUtils.__get_neighbours(*node)
                neighbours = [neighbour for neighbour in neighbours if
                              GraphUtils.__isvalid_coordinates(neighbour[0], neighbour[1], row, col)]
                neighbours = [neighbour for neighbour in neighbours if
                              not GraphUtils.__wall_among(world_matrix[neighbour[0]][neighbour[1]])]
                for neighbour in neighbours:
                    graph_dict[node][neighbour] = 1
        return graph_dict

    @staticmethod
    def __get_neighbours(x: int, y: int):
        return (x + 1, y), (x, y + 1), (x - 1, y), (x, y - 1)

    @staticmethod
    def __isvalid_coordinates(x: int, y: int, max_x: int, max_y: int) -> bool:
        return 0 <= x < max_x and 0 <= y < max_y

    @staticmethod
    def __wall_among(things):
        return bool([thing for thing in things if thing.__class__.__name__ == 'Wall'])
