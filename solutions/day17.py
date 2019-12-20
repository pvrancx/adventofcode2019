from enum import Enum
from typing import List, NamedTuple

import numpy as np

from src.intcode import IntComputer


class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    def turn_right(self):
        return Direction((self.value + 1) % 4)

    def turn_left(self):
        return Direction((self.value - 1) % 4)


class Location(NamedTuple):
    row: int
    column: int

    def move(self, direction: Direction):
        if direction is Direction.UP:
            return Location(self.row - 1, self.column)
        elif direction is Direction.DOWN:
            return Location(self.row + 1, self.column)
        elif direction is Direction.RIGHT:
            return Location(self.row, self.column + 1)
        elif direction is Direction.LEFT:
            return Location(self.row, self.column - 1)


def process_img(symbols: List[int]) -> np.ndarray:
    result = []
    line = []
    for d in symbols:
        if d == 10:
            if len(line) > 0:
                result.append(line.copy())
            line = []
        else:
            line.append(d)
    return np.array(result)


def find_intersections(img: np.ndarray):
    num_rows, num_cols = img.shape
    rows, columns = np.where(img == 35)
    result = []
    for r, c in zip(rows, columns):
        if (r == 0) or (c == 0) or (r == num_rows - 1) or (c == num_cols - 1):
            continue
        if img[r+1, c] == img[r-1, c] == img[r, c-1] == img[r, c+1] == 35:
            result.append([r, c])
    return np.array(result)


def calibration_code(coords) -> int:
    return np.sum(np.prod(coords, axis=-1))


def valid(loc, img) -> bool:
    n_rows, n_cols = img.shape
    return 0 <= loc.row < n_rows and 0 <= loc.column < n_cols


def move_line(img, start, direction):
    loc = start
    next_loc = loc.move(direction)
    while valid(next_loc, img) and img[next_loc.row, next_loc.column] == 35:
        loc = next_loc
        next_loc = loc.move(direction)
    return loc, np.abs(loc.row - start.row + loc.column - start.column)[0]


def follow_path(img):
    instructions = ['L']
    r, c = np.where(img == ord('^'))
    loc = Location(r, c)
    direction = Direction.LEFT

    while True:
        loc, steps = move_line(img, loc, direction)
        instructions.append(str(steps))
        left_move = loc.move(direction.turn_left())
        right_move = loc.move(direction.turn_right())
        if valid(left_move, img) and img[left_move.row, left_move.column] == 35:
            direction = direction.turn_left()
            instructions.append('L')
        elif valid(right_move, img) and img[right_move.row, right_move.column] == 35:
            direction = direction.turn_right()
            instructions.append('R')
        else:
            break
    return ','.join(instructions)


def compress_str(txt: str):
    code_dict = {}
    for idx, ch in enumerate(txt):
        code_dict[ch] = idx
    current_str = ''
    code = None
    result = []
    n_repeats = 10  # Hack: try to  repeat str to build dict and get better compression
    for idx, ch in enumerate(txt*n_repeats):
        if idx % len(txt) == 0:  # only return encoding for last repreat
            result.clear()
            current_str = ''

        current_str += ch
        if current_str in code_dict:
            code = code_dict[current_str]
        else:
            result.append(code)
            if len(current_str) < 20:
                code_dict[current_str] = len(code_dict)
            current_str = ch
    return result, code_dict


def encode_program(main_routine:str, sub_routines: List[str]) -> List[int]:
    result = []

    for routine in [main_routine] + sub_routines:
        for ch in routine:
            result.append(ord(ch))
        result.append(10)
    return result


if __name__ == '__main__':
    def _main():
        program = np.loadtxt('../inputs/day17.txt', delimiter=',', dtype=np.int64)
        computer = IntComputer(program.copy())
        computer.run(False)
        img = process_img(computer.output_stream.to_list())
        print(calibration_code(find_intersections(img)))
        instr = follow_path(img)
        # print(instr)
        # instr = instr.replace('L,10,L,12,R,6', 'A')
        # print(instr)
        # instr = instr.replace('R,10,L,4,L,4,L,12', 'B')
        # print(instr)
        # instr = instr.replace('L,10,R,10,R,6,L,4', 'C')
        # print(instr)
        # #print(instr.replace('', 'C'))
        inp = encode_program('A,B,A,B,A,C,B,C,A,C',
                             ['L,10,L,12,R,6',
                              'R,10,L,4,L,4,L,12',
                              'L,10,R,10,R,6,L,4'])
        program2 = program.copy()
        program2[0] = 2
        computer = IntComputer(program2)
        for i in inp:
            computer.input_stream.write(i)
        computer.input_stream.write(ord('n'))
        computer.input_stream.write(10)
        computer.run(False)
        print(computer.output_stream.to_list()[-1])


    _main()
