from collections import deque
from typing import NamedTuple, List, Dict, Tuple, Deque
import numpy as np


class Reagent(NamedTuple):
    quantity: int
    name: str


class Reaction(NamedTuple):
    reagents: List[Reagent]
    outcome: Reagent


def parse_reagent(txt: str) -> Reagent:
    list_str = txt.strip().split(' ')
    assert len(list_str) == 2
    return Reagent(int(list_str[0]), list_str[-1])


def parse_reagent_list(txt: str) -> List[Reagent]:
    list_str = txt.strip().split(',')
    return [parse_reagent(txt) for txt in list_str]


def parse_reaction(txt: str):
    list_str = txt.strip().split('=>')
    assert len(list_str) == 2
    reagents = parse_reagent_list(list_str[0])
    outcome = parse_reagent(list_str[-1])
    return Reaction(reagents, outcome)


def read_file(filename: str) -> Dict[str, Reaction]:
    with open(filename) as f:
        return read_reactions_str(f.read())


def read_reactions_str(txt: str) -> Dict[str, Reaction]:
    result = {}
    lines = txt.strip().splitlines()
    for line in lines:
        reaction = parse_reaction(line)
        result[reaction.outcome.name] = reaction
    return result


def process_reaction(reaction: Reaction, times: int) \
        -> Tuple[List[Reagent], int]:

    num_ores = 0
    list_of_reagents = []
    for reaction_input in reaction.reagents:
        if reaction_input.name == 'ORE':
            num_ores += times * reaction_input.quantity
        else:
            list_of_reagents.append(
                Reagent(
                    reaction_input.quantity * times,
                    reaction_input.name
                ))
    return list_of_reagents, num_ores


def determine_quantity(reagent: Reagent, inventory: Dict[str, Reagent]) -> int:
    if reagent.name not in inventory:
        return reagent.quantity
    else:
        available = inventory[reagent.name].quantity
        if reagent.quantity >= available:
            del inventory[reagent.name]
            return reagent.quantity - available
        else:
            available -= reagent.quantity
            inventory[reagent.name] = Reagent(available, reagent.name)
            return 0


def get_required_ores(target: Reagent, reactions: Dict[str, Reaction]) -> int:
    num_ores = 0
    reagents_to_produce = deque()
    reagents_to_produce.append(target)
    reagents_inventory = {}
    while len(reagents_to_produce) != 0:
        reagent = reagents_to_produce.popleft()
        reaction = reactions[reagent.name]
        needed = determine_quantity(reagent, reagents_inventory)
        # number of times we need to run reaction
        times = int(np.ceil(needed / reaction.outcome.quantity))
        # track surplus outcomes
        surplus = reaction.outcome.quantity * times - needed
        if surplus > 0:
            reagents_inventory[reagent.name] = Reagent(surplus, reagent.name)
        reagents_needed, ores_needed = process_reaction(reaction, times)
        num_ores += ores_needed
        reagents_to_produce.extend(reagents_needed)

    return num_ores


def get_max_fuel(num_ores: int, reactions: Dict[str, Reaction]) -> int:
    # determine search range
    min_ores = get_required_ores(Reagent(1, 'FUEL'), reactions)
    min_fuel = int(np.floor(num_ores / min_ores))  # can make at least this many
    max_fuel = min_fuel * 2
    max_ores = get_required_ores(Reagent(max_fuel, 'FUEL'), reactions)
    while max_ores < num_ores:  # find upper bound
        min_fuel = max_fuel
        max_fuel *= 2
        max_ores = get_required_ores(Reagent(max_fuel, 'FUEL'), reactions)

    # binary search
    while min_fuel <= max_fuel:
        mid_point = int(np.floor((min_fuel + max_fuel) / 2))
        mid_value = get_required_ores(Reagent(mid_point, 'FUEL'), reactions)
        if mid_value < num_ores:
            min_fuel = mid_point + 1
            min_ores = mid_value
        elif mid_value > num_ores:
            max_fuel = mid_point - 1
        else:
            return mid_point

    if min_ores > num_ores:
        return min_ores - 1
    else:
        return min_fuel


if __name__ == '__main__':
    def _main():
        reactions = read_file('../inputs/day14.txt')
        print(get_required_ores(Reagent(1, 'FUEL'), reactions))
        print(get_max_fuel(1000000000000, reactions))

    _main()

