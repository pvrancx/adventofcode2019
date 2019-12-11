from enum import Enum
from typing import NamedTuple, Dict

from src.intcode import IntComputer, InputDevice, OutputDevice
import numpy as np

import matplotlib.pyplot as plt


class Direction(Enum):
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3

    def turn_left(self):
        return Direction((self.value -1) %4)

    def turn_right(self):
        return Direction((self.value + 1) % 4)


class Position(NamedTuple):
    x: int
    y: int

    def move(self, direction: Direction):
        if direction is Direction.UP:
            return Position(self.x, self.y + 1)
        elif direction is Direction.DOWN:
            return Position(self.x, self.y - 1)
        elif direction is Direction.LEFT:
            return Position(self.x - 1, self.y)
        elif direction is Direction.RIGHT:
            return Position(self.x + 1, self.y)
        else:
            RuntimeError('Unknown direction')


class Robot(InputDevice, OutputDevice):
    def __init__(self):
        self._hull = {}  # type: Dict[Position, int]
        self._position = Position(0, 0)
        self._direction = Direction.UP
        self._paint = True

    def turn(self, dir: int):
        if dir == 1:
            self._direction = self._direction.turn_left()
        else:
            self._direction = self._direction.turn_right()

    def get_color(self) -> int:
        if self._position in self._hull:
            return self._hull[self._position]
        else:
            return 0

    def paint(self, color):
        if self.get_color() != color:
            self._hull[self._position] = color

    def move(self):
        self._position = self._position.move(self._direction)

    def ready(self):
        return True

    def has_output(self):
        return True

    def clear(self):
        pass

    def read(self):
        return self.get_color()

    def hull(self) -> Dict[Position, int]:
        return self._hull

    def write(self, instruction: int):
        if self._paint:
            self.paint(instruction)
        else:
            self.turn(instruction)
            self.move()
        self._paint = not self._paint


def run_robot(program: np.ndarray, start_color: int) -> Dict[Position, int]:
    computer = IntComputer(program)
    robot = Robot()
    robot.paint(start_color)
    computer.connect_input(robot)
    computer.connect_output(robot)
    computer.run(False)

    return robot.hull()


def show_hull(hull: Dict[Position, int]):
    coords = np.array([k for k in hull.keys()])
    minx, miny = np.min(coords, axis=0)
    maxx, maxy = np.max(coords, axis=0)

    grid = np.zeros((maxy-miny +1, maxx - minx +1))

    for pos, color in hull.items():
        grid[pos.y - miny, pos.x - minx] = color

    plt.matshow(np.flip(np.flip(grid, axis=1), axis=0))
    plt.show()


if __name__ == '__main__':
    def _main():
        inp = np.loadtxt('../inputs/day11.txt', delimiter=',').astype('int')
        hull = run_robot(inp.copy(), 0)
        print(len(hull))
        hull = run_robot(inp.copy(), 1)
        show_hull(hull)

    _main()
