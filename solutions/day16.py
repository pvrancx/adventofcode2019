import numpy as np


def create_pattern(repitition):
    base = [0, 1, 0, -1]
    return np.array([i for i in base for _ in range(repitition) ])


def conv(base_pattern, signal):
    idx = 0
    result = []
    while idx < signal.size:
        pattern = base_pattern[1:] if idx == 0 else base_pattern
        end = np.minimum(idx + pattern.size, signal.size)
        result.append(np.sum(signal[idx:end]*pattern[:(end-idx)]))
        idx += pattern.size

    return int(str(np.sum(result))[-1])


def do_phase(signal):
    return np.array([conv(create_pattern(idx+1), signal) for idx in range(signal.size)])


def fft(signal, n_phases):
    for _ in range(n_phases):
        signal = do_phase(signal)
        print(signal)
    return signal


def str_to_np(txt:str):
    return np.array([int(i) for i in txt])


if __name__ == '__main__':
    def _main():
        with open('../inputs/day16.txt') as f:
            inp = f.read()
        signal = inp.strip()
        print(fft(str_to_np(signal), 100)[:8])
    _main()
