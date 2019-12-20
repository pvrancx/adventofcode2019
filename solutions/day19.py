from typing import Tuple

from src.intcode import IntComputer
import numpy as np
import matplotlib.pyplot as plt


def approx_rc(computer: IntComputer, x_point: int) -> Tuple[float, float]:
    computer.reset()
    miny, maxy = get_yrange(computer, x_point)
    return -miny / x_point, -maxy / x_point,


def approx_solution(m1: float, m2: float) -> Tuple[float, float]:
    x1 = (-99. * m2 + 99.) / (m1 - m2)
    y1 = m1 * x1
    return int(np.floor(x1)), int(-np.ceil(y1))


def check_point(computer: IntComputer, x: int, y: int) -> bool:
    computer.reset()
    computer.input_stream.write(x)
    computer.input_stream.write(y)
    computer.run(False)
    return computer.output_stream.read() == 1


def get_yrange(computer, x_point: int) -> Tuple[int, int]:
    computer.reset()
    line = np.array([check_point(computer, x_point, y) for y in range(x_point)])
    idx = np.where(line == 1)[0]
    return np.min(idx), np.max(idx)


def get_grid(program: np.ndarray, maxx: int, maxy: int, minx=0, miny=0) -> np.ndarray:
    computer = IntComputer(program)
    grid = np.zeros((maxy, maxx))
    for x in range(minx, maxx):
        for y in range(miny, maxy):
            grid[y, x] = check_point(computer, x, y)
    return grid


def check_tractor(program: np.ndarray, maxx: int, maxy: int) -> int:
    total = 0
    for x in range(maxx):
        for y in range(maxy):
            computer = IntComputer(program)
            computer.input_stream.write(x)
            computer.input_stream.write(y)
            computer.run(False)
            total += computer.output_stream.read()
    return total


def valid_rect(computer, x, y):
    return check_point(computer, x+99, y) and check_point(computer, x, y + 99)


def search(computer, x0, y0):
    x, y = x0, y0
    while not check_point(computer, x + 99, y):
        y += 1
        while not check_point(computer, x, y + 99):
            x += 1
    return x, y


if __name__ == '__main__':
    def _main():
        program = np.loadtxt('../inputs/day19.txt', delimiter=',', dtype=np.int64)
        computer = IntComputer(program)
        m1, m2 = approx_rc(computer, 100)
        x1, y1 = approx_solution(m1, m2)
        print((x1-99, y1))

        x, y = search(computer, 0, 0)
        print(valid_rect(computer, x, y))
        print(x * 10000 + y)
        #
        # print((x2, y2))
        # print(check_point(computer, x2, y2))
        # print(valid_rect(computer, x2, int(y2)))
        #
        # grid = get_grid(program, 1140, 850, 1030, 736)
        # plt.matshow(grid)
        # plt.show()

        # x2, y2 = local_search(computer, int(626))
        # print(x2, y2)
        # print(get_yrange(computer, x1))
        # print(get_xrange(computer, y1))
        # # print(get_yrange(computer, x2-1))
        #
        # print(get_xrange(computer, y2))
        # miny, maxy = get_yrange(computer, x2+1)
        # print(miny, maxy)
        # print(get_xrange(computer, miny))


    _main()
