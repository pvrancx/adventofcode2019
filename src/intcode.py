import abc
from collections import deque
from enum import Enum
from typing import Tuple, List, NamedTuple

import numpy as np


class ParameterMode(Enum):
    POSITION = 0
    IMMEDIATE = 1
    RELATIVE = 2


class Parameter(NamedTuple):
    ptr: int
    mode: ParameterMode


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
    INC_REL_BASE = (9, 1, 0)

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


class InputDevice(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def read(self) -> int:
        raise NotImplementedError()

    @abc.abstractmethod
    def clear(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def ready(self) -> bool:
        raise NotImplementedError()


class OutputDevice(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def write(self, msg: int):
        raise NotImplementedError()

    @abc.abstractmethod
    def clear(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def has_output(self) -> bool:
        raise NotImplementedError()


class IoStream(InputDevice, OutputDevice):
    def __init__(self):
        self._stream = deque()

    def write(self, msg: int) -> int:
        self._stream.append(msg)

    def read(self) -> int:
        return self._stream.popleft()

    def clear(self):
        self._stream.clear()

    def ready(self):
        return len(self._stream) > 0

    def has_output(self):
        return len(self._stream) > 0


class IntComputer:
    def __init__(self, program: np.ndarray):
        self._program = program.copy()
        self._memory_ptr = 0
        self._memory = np.zeros(program.size * 2, dtype=np.int64)
        self._memory[0:program.size] = program[:]
        self._input_stream = IoStream()
        self._output_stream = IoStream()
        self._relative_base = 0

    def _set_mem_ptr(self, idx: int):
        assert -1 <= idx
        self._memory_ptr = idx

    def _inc_mem_ptr(self, val: int):
        assert -1 < self._memory_ptr + val
        self._memory_ptr += val

    def _consume_input(self):
        return self._input_stream.read()

    def _write_output(self, value: int):
        return self._output_stream.write(value)

    def _increase_memory(self):
        tmp = np.zeros(self._memory.size * 2, dtype=np.int64)
        tmp[0:self._memory.size] = self._memory[:]
        self._memory = tmp

    def _write_memory(self, parameter: Parameter, value: int):
        mode = parameter.mode
        assert mode is not ParameterMode.IMMEDIATE
        assert 0 <= parameter.ptr
        idx = self._memory[parameter.ptr]

        if mode is ParameterMode.RELATIVE:
            idx += self._relative_base

        assert idx >= 0
        while idx > self._memory.size:
            self._increase_memory()

        self._memory[idx] = value

    def _read_memory(self, parameter: Parameter) -> int:
        mode = parameter.mode
        assert 0 <= parameter.ptr
        idx = self._memory[parameter.ptr]
        if mode is ParameterMode.POSITION:
            assert 0 <= idx
            while idx > self._memory.size:
                self._increase_memory()
            return self._memory[idx]
        elif mode is ParameterMode.IMMEDIATE:
            return idx
        elif mode is ParameterMode.RELATIVE:
            rel_idx = self._relative_base + idx
            assert 0 <= rel_idx
            while rel_idx > self._memory.size:
                self._increase_memory()
            return self._memory[rel_idx]
        else:
            raise RuntimeError('Unknown parameter mode')

    def _read_params(self, opcode: OpCode, modes: List[ParameterMode]) -> List[Parameter]:
        params = []
        idx = self._memory_ptr + 1
        for param_id in range(opcode.num_params() + opcode.num_outputs()):
            ptr = idx + param_id
            mode = modes[param_id]
            params.append(Parameter(ptr, mode))
        return params

    def _parse_next_instruction(self) -> Tuple[OpCode, List[ParameterMode]]:
        instruction = self._memory[self._memory_ptr]
        int_str = str(instruction)
        opcode = OpCode(int(int_str[-2:]))
        param_modes = []
        for param_id in range(opcode.num_params() + opcode.num_outputs()):
            idx = -3 - param_id
            if param_id < len(int_str) - 2:
                param_modes.append(ParameterMode(int(int_str[idx])))
            else:
                param_modes.append(ParameterMode(0))

        return opcode, param_modes

    def reset(self):
        self._memory_ptr = 0
        self._memory = np.zeros_like(self._memory, dtype=np.int64)
        self._memory[0:self._program.size] = self._program[:]
        self._relative_base = 0
        self._output_stream.clear()

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

    def connect_input(self, input_stream: InputDevice):
        self._input_stream = input_stream

    def connect_output(self, output_stream: OutputDevice):
        self._output_stream = output_stream

    def has_output(self) -> bool:
        return self.output_stream.has_output()

    def has_input(self) -> bool:
        return self._input_stream.ready()

    def step(self, opcode: OpCode, args: List[Parameter]):
        if opcode is OpCode.HALT:
            self._set_mem_ptr(-1)
        elif opcode is OpCode.INPUT:
            target = args[-1]
            arg1 = self._consume_input()
            self._write_memory(target, arg1)
            self._inc_mem_ptr(2)
        elif opcode is OpCode.OUTPUT:
            arg1 = args[-1]
            value = self._read_memory(arg1)
            self._write_output(value)
            self._inc_mem_ptr(2)
        elif opcode is OpCode.ADD:
            arg1, arg2, target = args
            value = self._read_memory(arg1) + self._read_memory(arg2)
            self._write_memory(target, value)
            self._inc_mem_ptr(4)
        elif opcode is OpCode.MULTIPLY:
            arg1, arg2, target = args
            value = self._read_memory(arg1) * self._read_memory(arg2)
            self._write_memory(target, value)
            self._inc_mem_ptr(4)
        elif opcode is OpCode.JUMP_IF_TRUE:
            arg1, arg2 = args
            val1 = self._read_memory(arg1)
            val2 = self._read_memory(arg2)
            if val1 != 0:
                self._set_mem_ptr(val2)
            else:
                self._inc_mem_ptr(3)
        elif opcode is OpCode.JUMP_IF_FALSE:
            arg1, arg2 = args
            val1 = self._read_memory(arg1)
            val2 = self._read_memory(arg2)
            if val1 == 0:
                self._set_mem_ptr(val2)
            else:
                self._inc_mem_ptr(3)
        elif opcode == OpCode.LESS_THAN:
            arg1, arg2, target = args
            val1 = self._read_memory(arg1)
            val2 = self._read_memory(arg2)
            if val1 < val2:
                self._write_memory(target, 1)
            else:
                self._write_memory(target, 0)
            self._inc_mem_ptr(4)
        elif opcode == OpCode.EQUALS:
            arg1, arg2, target = args
            val1 = self._read_memory(arg1)
            val2 = self._read_memory(arg2)
            if val1 == val2:
                self._write_memory(target, 1)
            else:
                self._write_memory(target, 0)
            self._inc_mem_ptr(4)
        elif opcode == OpCode.INC_REL_BASE:
            arg = args[-1]
            self._relative_base += self._read_memory(arg)
            self._inc_mem_ptr(2)
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

