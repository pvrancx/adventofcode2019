import numpy as np


def create_pattern(repitition):
    base = [0, 1, 0, -1]
    return np.array([i for i in base for _ in range(repitition)])


def conv(base_pattern, signal):
    idx = 0
    result = []
    while idx < signal.size:
        pattern = base_pattern[1:] if idx == 0 else base_pattern
        end = np.minimum(idx + pattern.size, signal.size)
        result.append(np.sum(signal[idx:end]*pattern[:(end-idx)]))
        idx += pattern.size

    value = np.sum(result)
    return int(str(value)[-1])


def do_phase(signal):
    return np.array([conv(create_pattern(idx+1), signal) for idx in range(signal.size)])


def fft(signal, n_phases):
    for phase in range(n_phases):
        signal = do_phase(signal)
    return signal


def str_to_np(txt:str):
    return np.array([int(i) for i in txt])


def digits_to_number(digits):
    return int(''.join(str(i) for i in digits))


def compute(signal, offset, n_phases):
    # solution assumes offset >= len(signal)/2
    tail = signal[offset:]
    assert len(tail) < offset  # solution only works for 11111... part of pattern
    for _ in range(n_phases):
        tail = np.flip(np.flip(tail).cumsum()) % 10
    return tail[:8]


if __name__ == '__main__':
    def _main():
        with open("../inputs/day16.txt", "rt") as f:
            txt = f.read().strip()
        digits = str_to_np(txt)
        print(digits_to_number(fft(digits, 100)[:8]))
        offset = int(txt[:7])
        fft(digits, 100)
        print(digits_to_number(compute(digits.tolist()*10000, offset, 100)))


    _main()
