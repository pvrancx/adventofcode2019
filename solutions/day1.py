import numpy as np


def module_fuel(mass):
    return np.floor(mass / 3.) - 2.


def total_fuel(mass):
    total = 0
    fuel = module_fuel(mass)
    while fuel > 0:
        total += fuel
        fuel = module_fuel(fuel)
    return total


if __name__ == '__main__':
    inp = np.loadtxt('../inputs/day1.txt')
    print(np.sum([total_fuel(mod) for mod in inp]))