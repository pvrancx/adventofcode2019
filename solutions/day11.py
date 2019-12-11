from enum import Enum

from src.intcode import IntComputer
import numpy as np

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

class Direction(Enum):
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3

    def turn_left(self):
        return Direction((self.value -1) %4)

    def turn_right(self):
        return Direction((self.value + 1) % 4)


def get_color(hull, position):
    if position in hull:
        return hull[position]
    else:
        return 0


def run_robot(program: np.ndarray, grid):
    hull = {}
    computer = IntComputer(program)
    direction = Direction.UP
    position = Position(0, 0)
    while not computer.is_halted():
        computer.run(True)
        turn = computer.output_stream.popleft()
        paint_color = computer.output_stream.popleft()
        color = get_color(hull, position)
        position = position.move(direction)

    return hull



if __name__ == '__main__':
    print(Direction.LEFT.turn_left())
    print(Direction.LEFT.turn_right())
    print(Direction.DOWN.turn_right())
