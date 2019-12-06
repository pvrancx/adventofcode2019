import csv
from typing import List

import numpy as np


def manhattan_dist(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    assert xs.shape == ys.shape
    return np.abs(xs) + np.abs(ys)


def mark_wire(moves: List[str]) -> np.ndarray:
    position=np.zeros(2, dtype='int')
    points = [position.copy()]
    for move in moves:
        direction = move[0]
        steps = int(move[1:])
        new_pos = position.copy()
        if direction == 'R':
            new_pos[0] += steps
        elif direction == 'L':
            new_pos[0] -= steps
        elif direction == 'D':
            new_pos[1] -= steps
        elif direction == 'U':
            new_pos[1] += steps
        points.append(new_pos)
        position = new_pos
    return np.array(points, dtype='int')


def readfile(inputfile: str) -> List[str]:
    result = []
    with open(inputfile) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for line in reader:
            result.append(line)
    return result


def get_layout(wires):
    min_coords = np.array([np.inf, np.inf])
    max_coords= np.array([-np.inf, -np.inf])
    for coords in wires:
        min_coords = np.minimum(np.min(coords, axis=0), min_coords).astype(int)
        max_coords = np.maximum(np.max(coords, axis=0), max_coords).astype(int)
    grid_range = max_coords - min_coords
    return np.abs(min_coords[0]), np.abs(min_coords[1]), grid_range[0], grid_range[1]


def mark_grid(point_list, startx, starty, height, width):
    grid = np.zeros((width+1, height+1), dtype='bool')
    coords = point_list + np.array([[startx,starty]], dtype=int)
    point = coords[0]
    for p in coords[1:]:
        if point[0] == p[0]: #vertical move
            y1, y2 = (point[1], p[1]) if point[1] < p[1] else (p[1], point[1])
            grid[p[0], y1:y2] = True
        elif point[1] == p[1]: # horizontal move
            x1, x2 = (point[0], p[0]) if point[0] < p[0] else (p[0], point[0])
            grid[x1:x2, p[1]] = True
        else:
            RuntimeError('Cannot move diagonally')
        point = p
    return grid


def get_intersections(wires):
    points_lists = [mark_wire(wire) for wire in wires]
    startx, starty, width, height = get_layout(points_lists)
    grids = [mark_grid(coords, startx, starty, height, width) for coords in points_lists]

    result = grids[0]
    #result[startx, starty] =  False  #ignore start
    for grid in grids[1:]:
        result = np.logical_and(result, grid)
    return result


def get_min_distance(intersection_grid: np.ndarray) -> int:
    xs, ys = np.where(intersection_grid)
    dists = manhattan_dist(xs, ys)
    return np.min(dists)


def get_steps(coords, intersections: np.ndarray) -> int:
    point1 = coords[0]
    steps_so_far = 0
    xs, ys = np.where(intersections)
    int_steps = np.zeros_like(xs)
    for point2 in coords[1:]:
        int_points = []
        if point1[0] == point2[0]: #vertical move
            y1, y2 = (point1[1], point2[1]) if point1[1] < point2[1] else (point2[1], point1[1])
            int_points = np.where(np.logical_and(xs==point1[0], np.logical_and(y1<=ys, ys<= y2)))[0]
        elif point1[1] == point2[1]: # horizontal move
            x1, x2 = (point1[0], point2[0]) if point1[0] < point2[0] else (point2[0], point1[0])
            int_points = np.where(np.logical_and(ys==point1[1], np.logical_and(x1<= xs, xs <= x2)))[0]
        else:
            RuntimeError('Cannot move diagonally')

        for idx in int_points:
            point = np.array([xs[idx], ys[idx]])
            if int_steps[idx] == 0:
                int_steps[idx] = steps_so_far + np.sum(np.abs(point - point1))

        steps_so_far += np.sum(np.abs(point2 - point1))
        point1 = point2
    return int_steps


if __name__ =='__main__':
    def _main():
        inp = readfile('../inputs/day3.txt')
        intersections = get_intersections(inp)
        xs, ys = np.where(intersections)
        point_lists = [mark_wire(wire) for wire in inp]
        startx, starty, width, height = get_layout(point_lists)
        print('min distance')
        print(np.min(manhattan_dist(xs-startx, ys-starty)))

        all_steps = np.array([get_steps(coords+ np.array([[startx,starty]], dtype=int), intersections) for coords in point_lists])
        sum_steps = np.sum(all_steps, axis=0)
        print('min steps')
        print(np.min(sum_steps))

    _main()
