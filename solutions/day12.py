from typing import Tuple

import numpy as np


def get_gravity(coords: np.ndarray) -> np.ndarray:
    result = np.zeros_like(coords)
    for idx, coord in enumerate(coords):
        cmp = np.tile(coord, (coords.shape[0], 1))
        result[idx, ] = np.sum(-1 * (cmp > coords) + (cmp< coords), axis=0)
    return result


def apply_velocity(coords: np.ndarray, velocity: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    grav = get_gravity(coords)
    velocity += grav
    return coords + velocity, velocity


def simulate(
        coords: np.ndarray, velocity: np.ndarray, n_steps: int
) -> Tuple[np.ndarray, np.ndarray]:

    for _ in range(n_steps):
        coords, velocity = apply_velocity(coords, velocity)
    return coords, velocity


def potential_energy(pos: np.ndarray) -> float:
    return np.sum(np.abs(pos), axis=-1)


def kinetic_energy(vel: np.ndarray) -> float:
    return np.sum(np.abs(vel), axis=-1)


def find_period(pos: np.ndarray, vel: np.ndarray, dim: int) -> int:
    p = pos[:, dim, None]
    v = vel[:, dim, None]
    p0 = p.copy()
    steps = 0
    done = False
    while not done:
        p, v = apply_velocity(p, v)
        done = np.logical_and(
            np.all(p == p0),
            np.all(v == np.zeros_like(v))
        )
        steps += 1

    return steps


if __name__ == '__main__':
    def _main():
        inp = np.array([[-8, -18, 6],
                        [-11, -14, 4],
                        [8, -3, -10],
                        [-2, -16, 1]])
        pos, vel = simulate(inp.copy(), np.zeros_like(inp), 1000)
        print(np.sum(potential_energy(pos) * kinetic_energy(vel)))
        periods = [find_period(inp.copy(), np.zeros_like(inp), p) for p in range(3)]
        print(np.lcm.reduce(periods))

    _main()

