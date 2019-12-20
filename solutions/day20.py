import numpy as np

from src.graphs import Graph, Position


def valid_pos(pos, grid):
    n_rows, n_cols = grid.shape
    return 0 <= pos.x < n_cols and 0<= pos.y < n_rows


def neighbours(pos, grid):
    result = []
    neighb = [Position(pos.x+1, pos.y)



def build_graph(grid):
    graph = Graph()
    ys, xs = np.where(grid == ord('.'))
    for x, y in zip(xs, ys):
        graph.add_node(Position(x, y))

