import numpy as np

from src.intcode import IntComputer


def encode_input(txt: str, computer: IntComputer):
    inp = txt.strip()
    for ch in inp:
        computer.input_stream.write(ord(ch))
    computer.input_stream.write(ord('\n'))


def get_output(computer: IntComputer):
    computer.run(True)
    while not computer.is_halted() and computer.output_stream.to_list()[-1] != ord('\n'):
        computer.run(True)
    result = ''
    while computer.output_stream.ready():
        result += chr(computer.output_stream.read())
    return result


if __name__ == '__main__':
    def _main():
        program = np.loadtxt('../inputs/day25.txt', delimiter=',', dtype=np.int64)
        computer = IntComputer(program)
        while True:
            outp = get_output(computer)
            if outp.strip() == 'Command?':
                inp = input('Command? ')
                encode_input(inp, computer)
            else:
                print(outp[:-1])

    _main()

