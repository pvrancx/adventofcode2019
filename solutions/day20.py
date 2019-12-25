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


def get_neighbours(pos: Position, grid: np.ndarray) -> List[Position]:
    neighb = [Position(pos.x+1, pos.y),
              Position(pos.x, pos.y+1),
              Position(pos.x-1, pos.y),
              Position(pos.x, pos.y-1)]
    return [n for n in neighb if valid_pos(n, grid) and maze_pos(n, grid)]


def get_portal_id(p1: Position, p2: Position, grid: np.ndarray) -> str:
    return chr(grid[p1.y, p1.x]) + chr(grid[p2.y, p2.x])


def scan_h_portals(grid, portals) -> Dict[str, Tuple[Position, Position]]:
    for row_id in range(0, grid.shape[0], 2):
        row = grid[row_id]
        portal_idx = np.where(is_portal(row))[0]

        for col_id in portal_idx:
            up = Position(col_id, row_id-1)
            down = Position(col_id, row_id+1)
            if valid_pos(up, grid) and portal_pos(up, grid):
                portal_id = get_portal_id(up, Position(col_id, row_id), grid)
                if valid_pos(down, grid) and maze_pos(down, grid):
                    portal_entrance = down
                else:
                    portal_entrance = Position(col_id, row_id-2)
                if portal_id in portals:
                    portals[portal_id].append(portal_entrance)
                else:
                    portals[portal_id] = [portal_entrance]
            elif valid_pos(down, grid) and portal_pos(down, grid):
                portal_id = get_portal_id(Position(col_id, row_id), down, grid)
                if valid_pos(up, grid) and maze_pos(up, grid):
                    portal_entrance = up
                else:
                    portal_entrance = Position(col_id, row_id+2)
                if portal_id in portals:
                    portals[portal_id].append(portal_entrance)
                else:
                    portals[portal_id] = [portal_entrance]
    return portals


def scan_v_portals(grid: np.ndarray, portals: Dict) -> Dict[str, Tuple[Position, Position]]:
    for col_id in range(0, grid.shape[1], 2):
        col = grid[:, col_id]
        portal_idx = np.where(is_portal(col))[0]

        for row_id in portal_idx:
            left = Position(col_id-1, row_id)
            right = Position(col_id+1, row_id)
            if valid_pos(left, grid) and portal_pos(left, grid):
                portal_id = get_portal_id(left, Position(col_id, row_id), grid)
                if valid_pos(right, grid) and maze_pos(right, grid):
                    portal_entrance = right
                else:
                    portal_entrance = Position(col_id-2, row_id)
                if portal_id in portals:
                    portals[portal_id].append(portal_entrance)
                else:
                    portals[portal_id] = [portal_entrance]
            elif valid_pos(right, grid) and portal_pos(right, grid):
                portal_id = get_portal_id(Position(col_id, row_id), right, grid)
                if valid_pos(left, grid) and maze_pos(left, grid):
                    portal_entrance = left
                else:
                    portal_entrance = Position(col_id+2, row_id)
                if portal_id in portals:
                    portals[portal_id].append(portal_entrance)
                else:
                    portals[portal_id] = [portal_entrance]
    return portals


def get_portals(grid):
    portals = scan_h_portals(grid, {})
    return scan_v_portals(grid, portals)


def readmap(filename):
    result = []
    with open(filename, 'r') as f:
        for line in f:
            result.append([ord(ch) for ch in line])
    return np.array(result)


def build_graph(grid: np.ndarray) -> Tuple[Graph, Dict[str, Tuple[Position, Position]]]:
    graph = Graph()
    ys, xs = np.where(grid == ord('.'))
    portals = get_portals(grid)

    for x, y in zip(xs, ys):
        pos = Position(x, y)
        graph.add_node(pos)

    for pos in graph.get_nodes():
        neighbours = get_neighbours(pos, grid)
        for n in neighbours:
            graph.add_edge(pos, n)

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
    import os

    ffpath = os.path.join('C:/', 'Users','peter','Documents','ffmpeg','bin','ffmpeg.exe')
    print(ffpath)
    plt.rcParams['animation.ffmpeg_path'] = ffpath

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
            img =np.array(fig.canvas.renderer.buffer_rgba())
            video.append_data(img)


if __name__ == '__main__':
    def _main():
        grid = readmap('../inputs/day20.txt')
        graph, portals = build_graph(grid)

        start = portals['AA'][0]
        goal = portals['ZZ'][0]

        # result = dijkstra(graph, start, lambda n: n == goal)
        # print(result.cost)

        start_time = time.time()
        rgraph = RecursiveGraph(graph)

        result = dijkstra(rgraph, (0, start), lambda n: n == (0, goal))
        print(result.cost)
        # animate(grid,result.path)
        print(time.time() - start_time)

    _main()


