import pickle
from collections import deque
from typing import NamedTuple, Tuple

import numpy as np


class Position(NamedTuple):
    x: int
    y: int

    def get_id(self):
        return (self.x, self.y)


class SearchNode(NamedTuple):
    state: Tuple
    cost: int
    path: Tuple


def is_valid(pos, maze):
    maxy, maxx = maze.shape
    return (0 <= pos.x < maxx and
            0 <= pos.y < maxy and
            maze[pos.y, pos.x] != ord('#'))


def get_neighbours(pos,  maze):
    candidates = [Position(pos.x + 1, pos.y), Position(pos.x - 1, pos.y),
                  Position(pos.x, pos.y + 1),
                  Position(pos.x, pos.y - 1)]
    return [(n, 1) for n in candidates if is_valid(pos, maze)]


def get_key(door_value):
    return door_value + 32


def get_keys_needed(path, maze):
    result = []
    for pos in path:
        if is_door(maze[pos.y, pos.x]):
            result.append(get_key(maze[pos.y, pos.x]))
    return result


def bfs(start, maze, goal_fn, neigb_fn, return_on_goal=True):
    agenda = deque()
    agenda.append(SearchNode(start, 0, ()))
    visited = set()
    solution = None
    best_cost = np.inf
    while len(agenda) > 0:
        next_node = agenda.popleft()
        if goal_fn(next_node):
            print('Solution found - Cost: %d'%next_node.cost)
            if return_on_goal:
                return next_node
            elif next_node.cost < best_cost:
                best_cost = next_node.cost
                solution = next_node
        visited.add(next_node.state.get_id())
        neighb = neigb_fn(next_node.state, maze)
        for state, cost in neighb:
            if state.get_id() not in visited:
                agenda.append(SearchNode(state,
                                         next_node.cost + cost,
                                         next_node.path + (next_node.state,)))
    return solution


def readmap(filename):
    result = []
    with open(filename, 'r') as f:
        for line in f:
            result.append([ord(ch) for ch in line.strip()])
    return np.array(result)


def is_door(values):
    return np.logical_and(65 <= values, values <= 90)


def is_key(values):
    return np.logical_and(97 <= values, values <= 122)


def get_doors_and_keys(path, maze):
    vals = np.array([maze[n.y, n.x] for n in path])
    door_idx = np.where(is_door(vals))[0]
    key_idx = np.where(is_key(vals))[0]
    return tuple(vals[door_idx]), tuple(vals[key_idx])


def build_meta_graph(maze):
    start_coords = np.where(maze == ord('@'))
    key_coords = np.where(is_key(maze))
    poss = [Position(start_coords[1][0], start_coords[0][0])]
    for coords in zip(key_coords[0], key_coords[1]):
        poss.append(Position(coords[1], coords[0]))
    edges = {}
    for idx, source in enumerate(poss):
        for target in poss[idx+1:]:
            print((source, target))
            result = bfs(source, maze, lambda n: n[0] == target, get_neighbours)
            doors, keys = get_doors_and_keys(result.path + (target,), maze)
            keys += (maze[target.y, target.x],)
            edges[(source, target)] = (result.path, doors, keys, result.cost)
            edges[(target, source)] = (tuple(reversed(result.path)), doors, keys, result.cost)
    return poss, edges


def build_graph(maze):
    start_coords = np.where(maze == ord('@'))
    key_coords = np.where(is_key(maze))
    poss = [Position(start_coords[1][0], start_coords[0][0])]
    for coords in zip(key_coords[0], key_coords[1]):
        poss.append(Position(coords[1], coords[0]))
    edges = {}
    for idx, source in enumerate(poss):
        if source not in edges:
            edges[source] = []
        for target in poss[idx+1:]:
            print((source, target))
            if target not in edges:
                edges[target] = []
            result = bfs(source, maze, lambda n: n[0] == target, get_neighbours)
            doors, keys = get_doors_and_keys(result.path + (target,), maze)
            keys += (maze[target.y, target.x],)
            edges[source].append((target, result.path, doors, keys, result.cost))
            edges[target].append((source, tuple(reversed(result.path)), doors, keys, result.cost))
    return poss, edges


class MetaState(NamedTuple):
    pos: Position
    keys: Tuple

    def get_id(self):
        return self.pos + self.keys


class MultiRobotState(NamedTuple):
    positions: Tuple[Position]
    keys: Tuple

    def get_id(self):
        return self.positions + self.keys


def is_reachable(state, new_keys, doors):
    all_keys = state.keys + new_keys
    return np.all([get_key(door) in all_keys for door in doors])


def get_neighbours_meta(state: MetaState, graph):
    nodes, edges = graph
    result = []
    for pos in nodes:
        if pos == state.pos:
            continue
        _, doors, keys, cost = edges[(state.pos, pos)]
        if is_reachable(state, keys, doors):
            all_keys = set(state.keys + keys)
            result.append((MetaState(pos, tuple(all_keys)), cost))
    return result


def is_feasible(keys, doors):
    return np.all([get_key(door) in keys for door in doors])


def get_graph_neighbours(state, graph):
    pos = state.pos
    keys = state.keys
    nodes, edges = graph
    possible_edges = edges[pos]
    result = []
    for edge in possible_edges:
        target, _, doors, edge_keys, cost = edge
        all_keys = set(keys + edge_keys)
        if is_feasible(all_keys, doors):
            result.append((MetaState(target, tuple(all_keys)), cost))
    return result


def get_multi_robot_neigbours(state: MultiRobotState, graph):
    result = []
    for idx, robot_pos in enumerate(state.positions):
        sub_graph = graph[0][idx], graph[1][idx]
        all_positions = list(state.positions)
        robot_results = get_graph_neighbours(MetaState(robot_pos, state.keys), sub_graph)
        for new_robot_state, cost in robot_results:
            new_positions = all_positions.copy()
            new_positions[idx] = new_robot_state.pos
            new_keys = set(state.keys + new_robot_state.keys)
            if len(new_keys) > len(state.keys):
                result.append((MultiRobotState(tuple(new_positions), tuple(new_keys)), cost))
    return result


def build_multi_robot_graph(maze):
    split_maze = maze.copy()
    s = ord('@')
    w = ord('#')
    starty, startx = np.where(maze == s)
    new_pattern = np.array([[s, w, s],
                            [w, w, w],
                            [s, w, s]], dtype=int)
    split_maze[starty[0]-1:starty[0]+2, startx[0]-1:startx[0]+2] = new_pattern
    nodes = []
    edges = []
    sub_mazes = [split_maze[:starty[0]+1, :startx[0]+1],
                 split_maze[:starty[0]+1,startx[0]:],
                 split_maze[starty[0]:, :startx[0]+1],
                 split_maze[starty[0]:, startx[0]:]
                ]
    for sub_maze in sub_mazes:
        sub_n, sub_e = build_graph(sub_maze)
        nodes.append(sub_n)
        edges.append(sub_e)
    return split_maze, nodes, edges


def get_all_keys(maze):
    return maze[is_key(maze)]


def dijkstra(graph, start: MetaState, goal_fn, neigh_fn):
    agenda = []
    start_node = SearchNode(start, 0, ())
    agenda.append(start_node)
    dists = {start_node.state.get_id(): start_node}
    visited = set()

    while len(agenda) > 0:

        min_dist = np.inf
        next_node = None
        for n in agenda:
            if dists[n.state.get_id()].cost < min_dist and n.state.get_id() not in visited:
                min_dist = dists[n.state.get_id()].cost
                next_node = n
        agenda.remove(next_node)
        visited.add(next_node.state.get_id())
        if goal_fn(next_node):
            return next_node

        neighbours = neigh_fn(next_node.state, graph)

        for (state, weight) in neighbours:
            node_dist = dists[state.get_id()].cost if state.get_id() in dists else np.inf
            tot_dist = weight + min_dist
            if tot_dist < node_dist:
                node = SearchNode(
                    state,
                    tot_dist,
                    next_node.path + (next_node.state,))
                dists[state.get_id()] = node
                agenda.append(node)


if __name__ == '__main__':
    def _main():
        maze = readmap('../inputs/day18.txt')
        split_maze, nodes, edges = build_multi_robot_graph(maze)
        with open('mgraph.pkl', 'wb') as f:
              pickle.dump((nodes, edges), f)

        all_keys = tuple(np.sort(get_all_keys(maze)))
        goal_fn = lambda n: set(n.state.keys).issuperset(set(all_keys))

        start = np.where(split_maze == ord('@'))

        start_pos = (Position(x=39, y=39),
                     Position(x=1, y=39),
                     Position(x=39, y=1),
                     Position(x=1, y=1))

        result = dijkstra((nodes, edges),
                          MultiRobotState(start_pos, ()),
                          goal_fn,
                          get_multi_robot_neigbours)
        print(result.cost)

    _main()





