from src.intcode import IntComputer, IoStream
import numpy as np


class InputBuffer(IoStream):

    def __init__(self):
        super().__init__()
        self._empty_read = False

    def read(self):
        self._empty_read = False
        if not self.ready():
            self._empty_read = True
            return -1
        else:
            return super().read()

    def is_idle(self):
        return self._empty_read and (not self.ready())


def run_until_output(computer: IntComputer):
    result = []
    while len(result) < 3:
        computer.run(True)
        result += computer.output_stream.to_list()
        computer.output_stream.clear()
    return result


def _main():
    program = np.loadtxt('../inputs/day23.txt', delimiter=',', dtype=np.int64)
    computers = []
    for idx in range(50):
        computers.append(IntComputer(program.copy()))
        computers[-1].connect_input(InputBuffer())
        computers[-1].input_stream.write(idx)

    idx = 0
    nat_msg = []
    last_y = None
    idle_count = 0
    while True:
        computers[idx].process_next_instruction()
        if len(computers[idx].output_stream) == 3:
            output = computers[idx].output_stream.to_list()

            target = output[0]
            if target == 255:
                nat_msg = output.copy()
            else:
                computers[target].input_stream.write(output[1])
                computers[target].input_stream.write(output[2])

            computers[idx].output_stream.clear()

        if np.all([computer.input_stream.is_idle() for computer in computers]) \
                and np.all([len(computer.output_stream) == 0 for computer in computers]):
            idle_count += 1
        else:
            idle_count = 0

        if idle_count > 2000:
            print('idle')
            print(nat_msg)
            computers[0].input_stream.write(nat_msg[1])
            computers[0].input_stream.write(nat_msg[2])
            idle_count = 0

            if nat_msg[2] == last_y:
                print(nat_msg[2])
                break

            last_y = nat_msg[2]

        idx = (idx + 1) % 50


if __name__ == '__main__':
    _main()
