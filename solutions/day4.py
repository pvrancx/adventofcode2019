import numpy as np


def get_digits(n: int) -> np.ndarray:
    return np.array([int(d) for d in str(n)])


def check_double(digits: np.ndarray) -> bool:
    return np.any(digits[:-1] == digits[1:])


def check_seq(digits: np.ndarray) -> bool:
    d = digits[0]
    seq = 1
    for idx, next_digit in enumerate(digits[1:]):
        if d == next_digit:
            seq += 1
        else:
            if seq == 2:
                return True
            else:
                seq = 1
        d = next_digit
    return seq == 2


def part1_filter(n: int) -> bool:
    digits = get_digits(n)
    double_digits = check_double(digits)
    non_decreasing = np.all(digits[:-1] <= digits[1:])
    return double_digits and non_decreasing


def part2_filter(n: int) -> bool:
    digits = get_digits(n)
    double_digits = check_seq(digits)
    non_decreasing = np.all(digits[:-1] <= digits[1:])
    return double_digits and non_decreasing


def check_valid2(lower: int, upper: int) -> int:
    return len(list(filter(part1_filter, range(lower, upper+1))))


def check_valid1(lower: int, upper: int) -> int:
    return len(list(filter(part2_filter, range(lower, upper+1))))


if __name__ == '__main__':
    def _main():
        print(check_valid1(153517, 630395))
        print(check_valid2(153517, 630395))
    _main()
