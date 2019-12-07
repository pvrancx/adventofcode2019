import itertools
from collections import deque
from enum import Enum
from typing import Tuple, List

import numpy as np


class ParameterMode(Enum):
    POSITION = 0
    IMMEDIATE = 1


class OpCode(Enum):
    # (Opcode, n_in, n_out)
    HALT = (99, 0, 0)
    ADD = (1, 2, 1)
    MULTIPLY = (2, 2, 1)
    INPUT = (3, 0, 1)
    OUTPUT = (4, 1, 0)
    JUMPIFTRUE = (5, 2, 0)
    JUMPIFFALSE = (6, 2, 0)
    LESSTHAN = (7, 2, 1)
    EQUALS = (8, 2, 1)

    def __new__(cls, *args, **kwargs):
        obj = object.__new__(cls)
        obj._value_ = args[0]
        return obj

    def __init__(self, _: int, n_params: int, n_outs: int):
        self._nparams = n_params
        self._nouts = n_outs

    def num_params(self):
        return self._nparams

    def num_outputs(self):
        return self._nouts


def get_value(program: np.ndarray, code: int, mode: ParameterMode) -> int:
    if mode is ParameterMode.POSITION:
        return program[code]
    elif mode is ParameterMode.IMMEDIATE:
        return code
    else:
        raise RuntimeError('Unknown parameter mode')


def parse_instruction(instruction) -> Tuple[OpCode, List[ParameterMode]]:
    int_str = str(instruction)
    opcode = OpCode(int(int_str[-2:]))
    param_modes = []
    for idx in range(-3, -(len(int_str) + 1), -1):
        param_modes.append(ParameterMode(int(int_str[idx])))
    return opcode, param_modes


def get_param_mode(mode_list, idx):
    if idx >= len(mode_list):
        return ParameterMode(0)
    else:
        return mode_list[idx]


def get_params(program, opcode, start_idx, modes):
    params = []
    assert start_idx + opcode.num_params() + opcode.num_outputs() <= program.size
    for param_id in range(opcode.num_params()):
        code = program[start_idx + param_id]
        mode = get_param_mode(modes, param_id)
        params.append(get_value(program, code, mode))
    for out_id in range(opcode.num_outputs()):
        code = program[start_idx + opcode.num_params() + out_id]
        params.append(get_value(program, code, ParameterMode.IMMEDIATE))
    return params


def execute(program, input_stream, output_stream, start_ptr=0):
    ptr = start_ptr
    while ptr < program.size:

        opcode, param_modes = parse_instruction(program[ptr])
        params = get_params(program, opcode, ptr + 1, param_modes)

        if opcode is OpCode.HALT:
            return program, -1
        elif opcode is OpCode.INPUT:
            target = params[-1]
            arg1 = input_stream.popleft()
            program[target] = arg1
            ptr += 2
        elif opcode is OpCode.OUTPUT:
            arg1 = params[-1]
            output_stream.append(arg1)
            ptr += 2
            break
        elif opcode is OpCode.ADD:
            arg1, arg2, target = params
            program[target] = arg1 + arg2
            ptr += 4
        elif opcode is OpCode.MULTIPLY:
            arg1, arg2, target = params
            program[target] = arg1 * arg2
            ptr += 4
        elif opcode is OpCode.JUMPIFTRUE:
            arg1, arg2 = params
            if arg1 != 0:
                ptr = arg2
            else:
                ptr += 3
        elif opcode is OpCode.JUMPIFFALSE:
            arg1, arg2 = params
            if arg1 == 0:
                ptr = arg2
            else:
                ptr += 3
        elif opcode == OpCode.LESSTHAN:
            arg1, arg2, target = params
            if arg1 < arg2:
                program[target] = 1
            else:
                program[target] = 0
            ptr += 4
        elif opcode == OpCode.EQUALS:
            arg1, arg2, target = params
            if arg1 == arg2:
                program[target] = 1
            else:
                program[target] = 0
            ptr += 4
        else:
            raise RuntimeError('Unknown Opcode')
    return program, ptr


def run_sequence(program, settings):
    input_streams = [deque() for _ in range(len(settings))]
    output_streams = input_streams[1:] + [deque()]
    for idx, setting in enumerate(settings):
        input_streams[idx].append(setting)
    input_streams[0].append(0)
    for idx, setting in enumerate(settings):
        phase_program = program.copy()
        execute(phase_program, input_streams[idx], output_streams[idx])
    return output_streams[-1].popleft()


def run_continuous(program, settings):
    input_streams = [deque() for _ in range(len(settings))]
    output_streams = input_streams[1:] + [input_streams[0]]
    program_ptr = np.zeros(len(settings), dtype=np.int32)
    programs = [program.copy() for _ in range(len(settings))]
    for idx, setting in enumerate(settings):
        input_streams[idx].append(setting)
    input_streams[0].append(0)
    idx = 0
    while not np.all(program_ptr == -1):
        if program_ptr[idx] == -1:
            continue
        program, ptr = execute(programs[idx], input_streams[idx], output_streams[idx], program_ptr[idx])
        programs[idx] = program
        program_ptr[idx] = ptr
        idx = (idx +1) % len(settings)
    return output_streams[-1].popleft()


def find_sequence(program, min_value, max_value):
    best_value = -np.inf
    best_seq = []
    base_seq = list(range(min_value, max_value + 1))
    for seq in itertools.permutations(base_seq):
        value = run_sequence(program.copy(), seq)
        if value > best_value:
            best_value = value
            best_seq = seq
    return best_value


def find_continuous(program, min_value, max_value):
    best_value = -np.inf
    best_seq = []
    base_seq = list(range(min_value, max_value + 1))
    for seq in itertools.permutations(base_seq):
        value = run_continuous(program.copy(), seq)
        if value > best_value:
            best_value = value
            best_seq = seq
    return best_value


if __name__ == '__main__':
    def _main():
        inp1 = np.loadtxt('../inputs/day7.txt', delimiter=',').astype('int')
        print(find_sequence(inp1.copy(), 0, 4))

        print(find_continuous(inp1.copy(), 5, 9))


    _main()

