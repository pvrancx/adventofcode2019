from typing import Tuple

import numpy as np


def read_file(filename) -> str:
    with open(filename, 'r') as f:
        return f.read()


def str_to_np(txt: str) -> np.ndarray:
    result = []
    lines = txt.strip().split('\n')
    for line in lines:
        result.append([ch == '#' for ch in line.strip()])
    return np.array(result)


def get_polar_coords(xs: np.ndarray,
                     ys: np.ndarray,
                     x0: int,
                     y0: int) -> Tuple[np.ndarray, np.ndarray]:
    """polar coords with given location as origin """
    dxs = xs - x0
    dys = ys - y0
    rs = np.sqrt(dxs**2 + dys**2)
    angles = np.arctan2(dys, dxs)
    return rs, angles


def num_vis(grid: np.ndarray) -> Tuple[int, int, int]:
    ys, xs = np.where(grid)  # rows are ys
    result = np.zeros(ys.size, dtype=int)
    idx = 0
    for x, y in zip(xs,ys):
        _, angles = get_polar_coords(xs, ys, x, y)
        # only first asteroid at each angle can be seen
        result[idx] = np.unique(angles).size
        idx += 1
    best = np.argmax(result)
    return result[best], xs[best], ys[best]


def vaporize(grid: np.ndarray, x0: int, y0: int) -> Tuple[int, int]:
    grid_copy = grid.copy()
    # make sure we don't consider origin
    grid_copy[y0, x0] = False
    ys, xs = np.where(grid_copy)
    rs, angles = get_polar_coords(xs, ys, x0, y0)

    unique_angles = np.unique(angles)
    angle_idx = np.argmax(unique_angles > -np.pi/2) - 1  # negative y is up
    vaporised = np.zeros_like(rs, dtype=bool)
    num_vaporised = 0
    last_vaporised = ()

    while num_vaporised < 200:
        # loop over angles
        angle = unique_angles[angle_idx]
        # not vaporised asteroids at given angle
        asteroid_ids = np.logical_and(
            angles == angle,
            np.logical_not(vaporised))
        if np.any(asteroid_ids):
            # find closest asteroid at given angle
            r_temp = rs.copy()
            r_temp[np.logical_not(asteroid_ids)] = np.inf
            next_target = np.nanargmin(r_temp)
            last_vaporised = xs[next_target], ys[next_target]
            num_vaporised += 1
            print('Vaporised %d - asteroid at %d, %d'
                  % (num_vaporised,last_vaporised[0],last_vaporised[1]))
            vaporised[next_target] = True

        angle_idx= (angle_idx + 1) % unique_angles.size

    return last_vaporised


if __name__ == '__main__':
    def _main():
        grid = str_to_np(read_file('../inputs/day10.txt'))
        maxval, x0, y0 = num_vis(grid.copy())
        print(maxval)
        grid2 = str_to_np(read_file('../inputs/test.txt'))
        print(vaporize(grid.copy(), x0, y0))

    _main()

