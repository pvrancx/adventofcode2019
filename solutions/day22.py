import re
from enum import Enum
from typing import Tuple, List


class Action(Enum):
    DEAL_NEW = 0
    CUT = 1
    DEAL_WITH_INCR = 2


Instruction = Tuple[Action, int]


class Deck:
    def __init__(self, n_cards: int):
        self._cards = list(range(n_cards))
        self._n_cards = n_cards

    @property
    def num_cards(self):
        return self._n_cards

    def deal_new_stack(self):
        self._cards = list(reversed(self._cards))

    def cut(self, n_cards: int):
        cut, remainder = self._cards[:n_cards], self._cards[n_cards:]
        self._cards = remainder + cut

    def cards(self):
        return self._cards.copy()

    def deal_with_incr(self, incr: int):
        result = [None] * self.num_cards
        idx = 0
        while len(self._cards) > 0:
            result[idx] = self._cards.pop(0)
            idx = (idx + incr) % self.num_cards
        self._cards = result


def read_file(filename: str):
    result = []
    with open(filename, 'r') as f:
        for line in f:
            instr = re.findall("[a-z ]+", line)
            arg = re.findall("[\-0-9]+", line)
            if instr is []:
                continue
            elif instr[0].strip() == 'deal with increment':
                result.append((Action.DEAL_WITH_INCR, int(arg[-1])))
            elif instr[0].strip() == 'cut':
                result.append((Action.CUT, int(arg[-1])))
            elif instr[0].strip() == 'deal into new stack':
                result.append((Action.DEAL_NEW,0))
    return result


def shuffle(deck: Deck, instructions: List[Instruction]):
    for action, arg in instructions:
        if action is Action.DEAL_WITH_INCR:
            deck.deal_with_incr(arg)
        elif action is Action.CUT:
            deck.cut(arg)
        elif action is Action.DEAL_NEW:
            deck.deal_new_stack()
    return deck


def compute_idx(idx: int, n_cards: int, instructions: List[Instruction]) -> int:
    """Directly compute result of actions on index"""
    for action, arg in instructions:
        if action is Action.DEAL_WITH_INCR:
            idx = (idx * arg)
        elif action is Action.CUT:
            idx = (idx - arg)
        elif action is Action.DEAL_NEW:
            idx = (- idx - 1)
    return idx % n_cards


def compute_composite(n_cards: int, instructions: List[Instruction]) -> Tuple[int, int]:
    """ Compute coefficient and offset resulting from applying sequence of actions"""
    coeff = 1
    offset = 0
    for action, arg in instructions:
        if action is Action.DEAL_WITH_INCR:
            coeff *= arg
            offset *= arg
        elif action is Action.CUT:
            offset -= arg
        elif action is Action.DEAL_NEW:
            coeff *= -1
            offset *= -1
            offset -= 1
    return coeff % n_cards, offset % n_cards


def geometric(n: int, b: int, m:int) -> int:
    """ Compute nth partial sum of geometric series (1 + b + b**2 +b **3 + ... )mod m"""
    tmp = 1
    pow_b_mod_m = b % m
    total = 0
    while n > 0:
        if n & 1 == 1:
            total = (pow_b_mod_m * total + tmp) % m
        tmp = ((pow_b_mod_m + 1) * tmp) % m
        pow_b_mod_m = (pow_b_mod_m * pow_b_mod_m) % m
        n = n // 2
    return total


def compute_rep(repitions: int, n_cards: int, coeff:int, offset: int) -> Tuple[int, int]:
    """ Compute coeff and offset mod m resulting from repeatedly applying series of actions """
    rcoeff = pow(coeff, repitions, n_cards)  # modular exponentiation
    roffset = (geometric(repitions, coeff, n_cards) * offset) % n_cards
    return rcoeff, roffset


def egcd(a, b) -> Tuple[int, int, int]:
    """Extended euclidean algorithm -> compute  reciprocal mod n"""
    if a == 0:
        return (b, 0, 1)
    else:
        g, x, y = egcd(b % a, a)
        return (g, y - (b // a) * x, x)


if __name__ == '__main__':
    def _main():
        instr = read_file('../inputs/day22.txt')

        # result of actual shuffling
        print(shuffle(Deck(10007), instr).cards().index(2019))

        # compute index directly
        print(compute_idx(2019, 10007, instr))

        # compute by computing composite operation
        coeff, offset = compute_composite(10007, instr)
        print((2019*coeff + offset) % 10007)

        # test computing inverse operation
        inv = egcd(coeff, 10007)[1]
        result = ((6850 - offset) * inv) % 10007
        print(result)

        # test repeating shuffle & inv
        deck = Deck(10007)
        for _ in range(19):
            deck = shuffle(deck, instr)
        print(deck.cards()[1234])
        rcoeff, roffset = compute_rep(19, 10007, coeff, offset)

        rinv = egcd(rcoeff, 10007)[1]
        result = ((1234 - roffset) * rinv) % 10007
        print(result)

        coeff, offset = compute_composite(119315717514047, instr)
        rcoeff, roffset = compute_rep(101741582076661, 119315717514047, coeff, offset)
        rinv = egcd(rcoeff, 119315717514047)[1]
        result = ((2020 - roffset) * rinv) % 119315717514047
        print(result)

    _main()

