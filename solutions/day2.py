
import numpy as np


def process_code(program):
    idx = 0
    while idx < program.size:
        opcode = program[idx]
        f = None
        if opcode == 99:
            return program
        elif opcode == 1:
            f = lambda x, y: x+y
        elif opcode == 2:
            f = lambda x, y: x * y
        else:
            raise RuntimeError
        assert idx + 3 < program.size
        arg1 = program[idx+1]
        arg2 = program[idx+2]
        target = program[idx+3]
        program[target] = f(program[arg1], program[arg2])
        idx += 4
    return program


def reverse_engineer(program):
    for noun in range(100):
        for verb in range(100):
            test = program.copy()
            test[1] = noun
            test[2] = verb
            if process_code(test)[0] == 19690720:
                return noun, verb
    raise RuntimeError('Failed to find solution')


if __name__ == '__main__':
    def _main():
        inp1 = np.loadtxt('../inputs/day2.txt', delimiter=',').astype('int')
        inp2 = inp1.copy()
        inp1[1] = 12
        inp1[2] = 2
        print(process_code(inp1)[0])
        noun, verb = reverse_engineer(inp2)
        print(100*noun+verb)
    
    _main()
