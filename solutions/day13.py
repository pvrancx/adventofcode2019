from collections import deque

from src.intcode import OutputDevice, IntComputer, InputDevice
import numpy as np
import matplotlib.pyplot as plt


class Screen(OutputDevice):

    def __init__(self):
        self._scr = np.zeros((30, 40))
        self._buf = deque()
        self._score = 0
        plt.figure()
        self._ax = plt.gca()

    def _resize(self, new_width, new_height):
        height, width = self._scr.shape
        new_scrn = np.zeros(new_height, new_width)
        new_scrn[:height, :width] = self._scr[:,:]
        self._scr = new_scrn

    def _write_scrn(self, x, y, val):
        self._scr[y, x] = val
        self._ax.matshow(self._scr)
        plt.show(block=False)

    def write(self, msg: int):
        self._buf.append(msg)
        if len(self._buf) >= 3:
            x, y, v = self._buf.popleft(), self._buf.popleft(), self._buf.popleft()
            if x == -1 and y == 0:
                self._score = v
            else:
                self._write_scrn(x, y, v)

    def clear(self):
        self._buf.clear()

    def has_output(self):
        return len(self._buf) > 0


class JoyStick(InputDevice):
    def __init__(self):
        self._pos =0


if __name__ == '__main__':
    def _main():
        program = np.loadtxt('../inputs/day13.txt', delimiter=',', dtype=np.int64)
        computer = IntComputer(program.copy())
        screen = Screen()
        computer.connect_output(screen)
        computer.run(False)
        print(np.sum(screen._scr == 2))
        plt.show()
        program2 = program.copy()
        program2[0] = 1
        computer.set_program(program2)

    _main()



