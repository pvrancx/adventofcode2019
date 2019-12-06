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

    def apply(self, program, params, ptr):
        if self is OpCode.HALT:
            return ptr
        elif self is OpCode.ADD:
            arg1, arg2, target = params
            program[target] = arg1 + arg2
            return ptr + 4
        elif self is OpCode.MULTIPLY:
            arg1, arg2, target = params
            program[target] = arg1 * arg2
            return ptr + 4
        elif self is OpCode.INPUT:
            target = params[-1]
            program[target] = int(input('input: '))
            return ptr + 2
        elif self is OpCode.OUTPUT:
            arg1 = params[-1]
            print(arg1)
            return ptr + 2
        elif self is OpCode.JUMPIFTRUE:
            arg1, arg2 = params
            if arg1 != 0:
                return arg2
            else:
                return ptr + 3
        elif self is OpCode.JUMPIFFALSE:
            arg1, arg2 = params
            if arg1 == 0:
                return arg2
            else:
                return ptr + 3
        elif self == OpCode.LESSTHAN:
            arg1, arg2, target = params
            if arg1 < arg2:
                program[target] = 1
            else:
                program[target] = 0
            return ptr + 4
        elif self == OpCode.EQUALS:
            arg1, arg2, target = params
            if arg1 == arg2:
                program[target] = 1
            else:
                program[target] = 0
            return ptr + 4
        else:
            raise RuntimeError


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
    assert start_idx + opcode.num_params() + opcode.num_outputs() < program.size
    for param_id in range(opcode.num_params()):
        code = program[start_idx + param_id]
        mode = get_param_mode(modes, param_id)
        params.append(get_value(program, code, mode))
    for out_id in range(opcode.num_outputs()):
        code = program[start_idx + opcode.num_params() + out_id]
        params.append(get_value(program, code, ParameterMode.IMMEDIATE))
    return params


def execute(program):
    ptr = 0
    while ptr < program.size:
        opcode, param_modes = parse_instruction(program[ptr])
        if opcode is OpCode.HALT:
            return program
        else:
            params = get_params(program, opcode, ptr+1, param_modes)
            ptr = opcode.apply(program, params, ptr)
    return program


if __name__ == '__main__':
    def _main():
        inp1 = np.loadtxt('../inputs/day5.txt', delimiter=',').astype('int')
        execute(inp1)

    _main()

