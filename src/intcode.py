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
    JUMP_IF_TRUE = (5, 2, 0)
    JUMP_IF_FALSE = (6, 2, 0)
    LESS_THAN = (7, 2, 1)
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


class IntComputer:
    def __init__(self, program: np.ndarray):
        self._program = program.copy()
        self._memory_ptr = 0
        self._memory = program.copy()
        self._input_stream = deque()
        self._output_stream = deque()

    def _set_mem_ptr(self, idx: int):
        assert -1 <= idx < self._memory.size
        self._memory_ptr = idx

    def _inc_mem_ptr(self, val: int):
        assert -1 < self._memory_ptr + val < self._memory.size
        self._memory_ptr += val

    def _consume_input(self):
        return self._input_stream.popleft()

    def _write_output(self, value: int):
        return self._output_stream.append(value)

    def _write_memory(self, idx: int, value: int):
        assert 0 <= idx < self._memory.size
        self._memory[idx] = value

    def _read_memory(self, idx: int, mode: ParameterMode) -> int:
        if mode is ParameterMode.POSITION:
            assert 0 <= idx < self._memory.size
            return self._memory[idx]
        elif mode is ParameterMode.IMMEDIATE:
            return idx
        else:
            raise RuntimeError('Unknown parameter mode')

    def _read_params(self, opcode: OpCode, modes: List[ParameterMode]) -> List[int]:
        params = []
        idx = self._memory_ptr + 1
        for param_id in range(opcode.num_params()):
            code = self._memory[idx + param_id]
            mode = modes[param_id]
            params.append(self._read_memory(code, mode))
        return params

    def _read_target(self, opcode: OpCode) -> int:
        idx = self._memory_ptr + opcode.num_params() + 1
        return self._read_memory(idx, ParameterMode.POSITION)

    def _parse_next_instruction(self) -> Tuple[OpCode, List[ParameterMode]]:
        instruction = self._memory[self._memory_ptr]
        int_str = str(instruction)
        opcode = OpCode(int(int_str[-2:]))
        param_modes = []
        for param_id in range(opcode.num_params()):
            idx = -3 - param_id
            if param_id < len(int_str) - 2:
                param_modes.append(ParameterMode(int(int_str[idx])))
            else:
                param_modes.append(ParameterMode(0))
        return opcode, param_modes

    def reset(self):
        self._memory_ptr = 0
        self._memory = self._program.copy()

    def set_program(self, program: np.ndarray):
        self._program = program.copy()
        self.reset()

    @property
    def memory(self):
        return self._memory.copy()

    @property
    def input_stream(self):
        return self._input_stream

    @property
    def output_stream(self):
        return self._output_stream

    def connect_input(self, input_stream: deque):
        self._input_stream = input_stream

    def connect_output(self, output_stream: deque):
        self._output_stream = output_stream

    def has_output(self) -> bool:
        return len(self._output_stream) > 0

    def has_input(self) -> bool:
        return len(self._input_stream) > 0

    def step(self, opcode: OpCode, args: List[int]):
        if opcode is OpCode.HALT:
            self._set_mem_ptr(-1)
        elif opcode is OpCode.INPUT:
            target = self._read_target(opcode)
            arg1 = self._consume_input()
            self._write_memory(target, arg1)
            self._inc_mem_ptr(2)
        elif opcode is OpCode.OUTPUT:
            arg1 = args[-1]
            self._write_output(arg1)
            self._inc_mem_ptr(2)
        elif opcode is OpCode.ADD:
            target = self._read_target(opcode)
            arg1, arg2 = args
            self._write_memory(target, arg1 + arg2)
            self._inc_mem_ptr(4)
        elif opcode is OpCode.MULTIPLY:
            target = self._read_target(opcode)
            arg1, arg2 = args
            self._write_memory(target, arg1 * arg2)
            self._inc_mem_ptr(4)
        elif opcode is OpCode.JUMP_IF_TRUE:
            arg1, arg2 = args
            if arg1 != 0:
                self._set_mem_ptr(arg2)
            else:
                self._inc_mem_ptr(3)
        elif opcode is OpCode.JUMP_IF_FALSE:
            arg1, arg2 = args
            if arg1 == 0:
                self._set_mem_ptr(arg2)
            else:
                self._inc_mem_ptr(3)
        elif opcode == OpCode.LESS_THAN:
            target = self._read_target(opcode)
            arg1, arg2 = args
            if arg1 < arg2:
                self._write_memory(target, 1)
            else:
                self._write_memory(target, 0)
            self._inc_mem_ptr(4)
        elif opcode == OpCode.EQUALS:
            target = self._read_target(opcode)
            arg1, arg2 = args
            if arg1 == arg2:
                self._write_memory(target, 1)
            else:
                self._write_memory(target, 0)
            self._inc_mem_ptr(4)
        else:
            raise RuntimeError('Unknown OpCode')

    def is_halted(self) -> bool:
        return self._memory_ptr == -1

    def run(self, pause_on_output: bool = True):
        while not self.is_halted():
            opcode, param_modes = self._parse_next_instruction()
            params = self._read_params(opcode, param_modes)
            self.step(opcode, params)
            if pause_on_output and self.has_output():
                break

