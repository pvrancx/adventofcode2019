import numpy as np


def get_gravity(coords):
    result = np.zeros_like(coords)
    for idx, coord in enumerate(coords):
        cmp = np.tile(coord, (coords.shape[0], 1))
        result[idx, ] = np.sum(-1 * (cmp > coords) + (cmp< coords), axis=0)
    return result

def apply_velocity(coords, velocity):
    grav = get_gravity(coords)
    velocity += grav
    return coords + velocity, velocity


def simulate(coords, velocity, n_steps):
    for _ in range(n_steps):
        print(coords)
        print(velocity)
        coords, velocity = apply_velocity(coords, velocity)
    return coords, velocity


def potential_energy(pos):
    return np.sum(np.abs(pos), axis=-1)


def kinetic_energy(vel):
    return np.sum(np.abs(vel), axis=-1)


if __name__ == '__main__':
    def _main():
        inp = np.array([[-8, -18, 6],
                        [-11, -14, 4],
                        [8, -3, -10],
                        [-2, -16, 1]])
        pos, vel = simulate(inp, np.zeros_like(inp), 1000)
        print(np.sum(potential_energy(pos) * kinetic_energy(vel)))
    _main()

