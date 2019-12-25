import numpy as np
import matplotlib.pyplot as plt


def get_neighbours(world):
    neighbours = np.zeros_like(world, dtype=int)
    neighbours[1:] += world[:-1]  # North
    neighbours[:-1] += world[1:]  # South
    neighbours[:, 1:] += world[:, :-1]  # West
    neighbours[:, :-1] += world[:, 1:]  # East

    return neighbours


def update(world, neighbours):
    bug = np.logical_and(
        np.logical_not(world), # empty space
        np.logical_or(
            neighbours == 1,
            neighbours == 2
        )
    )

    empty = np.logical_and(
        world,  # bug space
        np.logical_not(neighbours == 1)
    )

    world[bug] = True
    world[empty] = False


def step(world: np.ndarray):
    neighbours = get_neighbours(world)
    update(world, neighbours)



def recursive_run(world, n_steps):
    levels = {0: world}
    min_level, max_level = 0, 0

    for step in range(n_steps):
        neighb_dct = {}
        min_level -= 1
        max_level += 1
        levels[min_level] = np.zeros_like(world)
        levels[max_level] = np.zeros_like(world)
        neighb_dct[min_level] = np.zeros_like(world, dtype=int)
        neighb_dct[max_level] = np.zeros_like(world, dtype=int)

        for level_id in range(min_level, max_level +1):
            level = levels[level_id]
            level[2, 2] = False

            neighbours = get_neighbours(level)

            if level_id > min_level:
                level_lower = levels[level_id - 1]
                neighbours[0, :] += level_lower[1, 2]
                neighbours[-1, :] += level_lower[3, 2]
                neighbours[:, 0] += level_lower[2, 1]
                neighbours[:, -1] += level_lower[2, 3]

            if level_id < max_level:
                level_higher = levels[level_id + 1]
                neighbours[1, 2] += np.sum(level_higher[0, :])
                neighbours[3, 2] += np.sum(level_higher[-1, :])
                neighbours[2, 1] += np.sum(level_higher[:, 0])
                neighbours[2, 3] += np.sum(level_higher[:, -1])

            neighb_dct[level_id] = neighbours

        for level_id in range(min_level, max_level+1):
            update(levels[level_id], neighb_dct[level_id])

    total = 0
    for level_id, level in levels.items():
        print(level_id)
        print(level)
        level[2,2] = False
        total += np.sum(level)
    return total


def readmap(filename):
    result = []
    with open(filename, 'r') as f:
        for line in f:
            result.append([ord(ch) for ch in line if ch != '\n'])
    result =  np.array(result)
    return result == ord('#')


if __name__ == '__main__':
    def _main():
        world = readmap('../inputs/day24.txt')


        confs = set()

        while True:
            step(world)
            conf = tuple(world.flatten().tolist())
            if conf in confs:
                print(np.sum(2**np.arange(25) * world.flatten()))
                break
            confs.add(conf)

        world = readmap('../inputs/day24.txt')
        print(recursive_run(world, 200))

    _main()
