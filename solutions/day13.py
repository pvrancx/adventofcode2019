import time
from collections import deque

import pygame

from src.intcode import OutputDevice, IntComputer, InputDevice
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def gray(im):
    im = 255 * (im / im.max())
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 2] = ret[:, :, 1] = ret[:, :, 0] = im
    return ret


class Screen(OutputDevice):

    def __init__(self):
        self._scr = np.zeros((30, 40))
        self._buf = deque()
        self._score = 0
        pygame.init()
        self._display = pygame.display.set_mode((350, 400))
        self._display.fill((0, 0, 0))

    def _resize(self, new_width, new_height):
        height, width = self._scr.shape
        new_scrn = np.zeros(new_height, new_width)
        new_scrn[:height, :width] = self._scr[:,:]
        self._scr = new_scrn

    def ball_pos(self):
        return np.where(self._scr == 4)

    def paddle_pos(self):
        return np.where(self._scr == 3)

    def _write_scrn(self, x, y, val):
        self._scr[y, x] = val
        if val == 4:
            self._update_scrn()

    def _update_scrn(self):
        self._display.fill((0, 0, 0))
        surf = pygame.surfarray.make_surface(gray(self._scr.copy().T))
        surf = pygame.transform.scale(surf, (300, 400))
        self._display.blit(surf, (0, 0))

        pygame.display.update()
        pygame.event.pump()
        time.sleep(0.05)

    def write(self, msg: int):
        self._buf.append(msg)
        if len(self._buf) >= 3:
            x, y, v = self._buf.popleft(), self._buf.popleft(), self._buf.popleft()
            if x == -1 and y == 0:
                print('SCORE: %d'%v)
                self._score = v
            else:
                self._write_scrn(x, y, v)

    def clear(self):
        self._buf.clear()

    def has_output(self):
        return len(self._buf) > 0


class JoyStick(InputDevice):
    def __init__(self):
        self._pos = 0

    def read(self):
        key = None
        while key is None:
            key = 0
            time.sleep(0.5)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        key = -1
                    if event.key == pygame.K_RIGHT:
                        key = 1

        return key

    def ready(self):
        return True

    def clear(self):
        pass


class AiJoyStick(InputDevice):
    def __init__(self, screen):
        self._screen = screen
        self._last_ball_x = 0

    def read(self):
        ball_y, ball_x = self._screen.ball_pos()
        paddle_y, paddle_x = self._screen.paddle_pos()

        print((ball_x, ball_y))
        print(paddle_x)

        if ball_x < paddle_x:
            self._last_ball_x = ball_x
            return -1
        elif ball_x > paddle_x:
            self._last_ball_x = ball_x
            return 1
        else:
            return 0

    def ready(self):
        return True

    def clear(self):
        pass


if __name__ == '__main__':
    def _main():
        program = np.loadtxt('../inputs/day13.txt', delimiter=',', dtype=np.int64)
        #computer = IntComputer(program.copy())

        # computer.run(False)
        # print(np.sum(screen._scr == 2))
        # running = True
        # while running:
        #     for event in pygame.event.get():
        #         if event.type == pygame.QUIT:
        #             running = False
        #     screen._update_scrn()
        program2 = program.copy()
        program2[0] = 2
        computer = IntComputer(program2.copy())
        screen = Screen()
        computer.connect_output(screen)
        #
        computer.set_program(program2)
        #computer.connect_input(JoyStick())

        computer.connect_input(AiJoyStick(screen))
        computer.run(False)

        print(screen._score)
        running = True
        while running:
             for event in pygame.event.get():
                 if event.type == pygame.QUIT:
                     running = False
             screen._update_scrn()

    _main()



