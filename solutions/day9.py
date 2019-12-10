import numpy as np

from src.intcode import IntComputer


if __name__ == '__main__':
    def _main():
        program = np.loadtxt('../inputs/day9.txt', delimiter=',', dtype=np.int64)
        computer = IntComputer(program)
        computer.input_stream.append(1)
        computer.run(False)
        print(computer.output_stream)
        computer.reset()
        computer.input_stream.append(2)
        computer.run(False)
        print(computer.output_stream)

    _main()
