import time
from typing import Tuple, Dict, List

import imageio
import numpy as np

from src.graphs import Graph, Position, dijkstra, Edge, AbstractGraph


def valid_pos(pos: Position, grid: np.ndarray) -> bool:
    n_rows, n_cols = grid.shape
    return 0 <= pos.x < n_cols and 0 <= pos.y < n_rows


def portal_pos(pos: Position, grid: np.ndarray) -> bool:
    return 65 <= grid[pos.y, pos.x] <= 90


def maze_pos(pos: Position, grid: np.ndarray) -> bool:
    return grid[pos.y, pos.x] == ord('.')


def is_portal(grid: np.ndarray) -> np.ndarray:
    return np.logical_and(grid >= 65, grid <= 90)


def get_neighbours(pos: Position, steps: int=1) -> List[Position]:
    neighb = [Position(pos.x+steps, pos.y),
              Position(pos.x, pos.y+steps),
              Position(pos.x-steps, pos.y),
              Position(pos.x, pos.y-steps)]
    return neighb


def get_portal_id(p1: Position, p2: Position, grid: np.ndarray) -> str:
    return chr(grid[p1.y, p1.x]) + chr(grid[p2.y, p2.x])


def readmap(filename):
    result = []
    with open(filename, 'r') as f:
        for line in f:
            result.append([ord(ch) for ch in line])
    return np.array(result)


def build_graph(grid: np.ndarray) -> Tuple[Graph, Dict[str, List[Position]]]:
    graph = Graph()
    ys, xs = np.where(grid == ord('.'))
    portals = {}

    for x, y in zip(xs, ys):
        pos = Position(x, y)
        graph.add_node(pos)

    for pos in graph.get_nodes():
        neighbours = get_neighbours(pos, steps=1)
        neighbours_2_steps = get_neighbours(pos, steps=2)
        for n, n2 in zip(neighbours, neighbours_2_steps):
            if maze_pos(n, grid):
                graph.add_edge(pos, n)
            if portal_pos(n, grid) and portal_pos(n2, grid):
                label = get_portal_id(n2, n, grid) if n2.y < n.y or n2.x < n.x else get_portal_id(n, n2, grid)
                if label in portals:
                    portals[label].append(pos)
                else:
                    portals[label] = [pos]

    for p_id, route in portals.items():
        if p_id not in ['AA', 'ZZ']:
            graph.add_edge(route[0], route[1], info='portal')
            graph.add_edge(route[1], route[0], info='portal')

    return graph, portals


class RecursiveGraph(AbstractGraph):
    def __init__(self, graph: Graph):
        self._graph = graph
        coords = np.array([[n.x, n.y] for n in graph.get_nodes()])
        self.minx, self.miny = np.min(coords, axis=0)
        self.maxx, self.maxy = np.max(coords, axis=0)

    def _is_outer_portal(self, pos):
        return pos.x == self.minx or \
               pos.x == self.maxx or \
               pos.y == self.miny or \
               pos.y == self.maxy

    def get_edges(self, node):

        level, pos = node
        result = []

        edges = self._graph.get_edges(pos)
        for edge in edges:
            if edge.info == 'portal':
                if self._is_outer_portal(pos):
                    new_level = level - 1
                else:
                    new_level = level + 1
                if new_level >= 0:
                    result.append(Edge(node, (new_level, edge.target), edge.weight, edge.info))
            else:
                result.append(Edge(node, (level, edge.target), edge.weight, edge.info))
        return result


def animate(grid, path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    n_rows, n_cols = grid.shape
    l_path = len(path)
    with imageio.get_writer('anim2.mp4', fps=30) as video:
        for idx, node in enumerate(path):
            print('%d / %d'%(idx, l_path))
            level, pos = node
            state = grid.copy()
            state[pos.y, pos.x] = 100
            ax.clear()
            ax.matshow(state, animated=True)
            ax.text(n_rows//2, n_cols//2, 'level %d'%level, color='blue')
            fig.canvas.draw()
            img = np.array(fig.canvas.renderer.buffer_rgba())
            video.append_data(img)


if __name__ == '__main__':
    def _main():
        grid = readmap('../inputs/day20.txt')
        graph, portals = build_graph(grid)

        start = portals['AA'][0]
        goal = portals['ZZ'][0]

        result = dijkstra(graph, start, lambda n: n == goal)
        print(result.cost)

        start_time = time.time()
        rgraph = RecursiveGraph(graph)

        result = dijkstra(rgraph, (0, start), lambda n: n == (0, goal))
        print(result.cost)
        # animate(grid,result.path)
        print(time.time() - start_time)

    _main()


