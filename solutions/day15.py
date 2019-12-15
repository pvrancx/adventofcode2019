from enum import Enum
from typing import NamedTuple, List, Set, Tuple, Union

import numpy as np

from src.intcode import IntComputer


class Movement(Enum):
    NORTH = 1
    SOUTH = 2
    WEST = 3
    EAST = 4

    def reverse_move(self):
        if self is Movement.NORTH:
            return Movement.SOUTH
        elif self is Movement.SOUTH:
            return Movement.NORTH
        elif self is Movement.EAST:
            return Movement.WEST
        elif self is Movement.WEST:
            return Movement.EAST
        else:
            raise RuntimeError()


class Status(Enum):
    FAIL = 0
    SUCCESS = 1
    GOAL = 2


class Position(NamedTuple):
    x: int
    y: int

    def apply_move(self, move: Movement):
        if move is Movement.NORTH:
            return Position(self.x, self.y + 1)
        elif move is Movement.SOUTH:
            return Position(self.x, self.y - 1)
        elif move is Movement.WEST:
            return Position(self.x - 1, self.y)
        elif move is Movement.EAST:
            return Position(self.x + 1, self.y)
        else:
            raise RuntimeError('Invalid move')


def expand_node(node: Position, valid_nodes: Set[Position]) -> List[Position]:
    return [node.apply_move(move) for move in Movement if node.apply_move(move) in valid_nodes]


def extend_frontier(
        frontier: Set[Tuple[Position, Tuple]],
        node: Position,
        path: Union[Tuple[Position], Tuple],
        known_nodes: Set[Position]):

    for move in Movement:
        neighb = node.apply_move(move)
        if neighb not in known_nodes:
            new_path = path + (move, )
            frontier.add((neighb, new_path))


def explore_map(program: np.ndarray, start: Position, find_goal: bool=True) \
        -> Tuple[Position, Tuple[Position], Set[Position]]:

    frontier = set()  # nodes to visit
    start = start
    expanded = set()  # visited nodes (including walls)
    valid_nodes = set()  # non-wall nodes

    expanded.add(start)
    valid_nodes.add(start)
    extend_frontier(frontier, start, (), expanded)

    goal_node = None
    goal_commands = ()

    while len(frontier) > 0:
        # next node to visit and path to node
        goal, commands = frontier.pop()

        expanded.add(goal)

        computer = IntComputer(program.copy())
        # add path to input
        for command in commands:
            computer.input_stream.write(command.value)

        output = -1
        while computer.input_stream.ready():
            computer.run(True)
            output = computer.output_stream.read()

        # status for goal node
        status = Status(output)

        if status is not Status.FAIL:
            extend_frontier(frontier, goal, commands, expanded)
            valid_nodes.add(goal)

        if status is Status.GOAL:
            goal_node = goal
            goal_commands = commands
            if find_goal:
                return goal_node, goal_commands, valid_nodes

    return goal_node, goal_commands, valid_nodes


def visit_all(start: Position, nodes: Set[Position]) -> int:
    visited = set()
    visited.add(start)
    steps = 0

    while len(visited) < len(nodes):
        neighbours = set()
        for node in visited:
            neighbours.update(set(expand_node(node, nodes)))
        visited.update(set(neighbours))
        steps += 1
    return steps


if __name__ == '__main__':
    def _main():

        program = np.loadtxt('../inputs/day15.txt', delimiter=',', dtype=np.int64)
        goal, path, nodes = explore_map(program, Position(0, 0), False)
        print(len(path))
        steps = visit_all(goal, nodes)
        print(steps)

    _main()

