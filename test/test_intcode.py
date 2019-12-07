import numpy as np

from src.intcode import IntComputer
import pytest


day_2_test_data = \
    [
        (np.array([1, 0, 0, 0, 99], dtype=np.int32), np.array([2, 0, 0, 0, 99], dtype=np.int32)),
        (np.array([2, 3, 0, 3, 99], dtype=np.int32), np.array([2, 3, 0, 6, 99], dtype=np.int32)),
        (np.array([2, 4, 4, 5, 99, 0], dtype=np.int32), np.array([2, 4, 4, 5, 99, 9801], dtype=np.int32)),
        (np.array([1, 1, 1, 4, 99, 5, 6, 0, 99], dtype=np.int32),
         np.array([30, 1, 1, 4, 2, 5, 6, 0, 99], dtype=np.int32))
    ]


# jump tests
day_5_program1 = np.array([3, 12, 6, 12, 15, 1, 13, 14, 13, 4, 13, 99, -1, 0, 1, 9], dtype=np.int32)

day_5_program2 = np.array([3, 3, 1105, -1, 9, 1101, 0, 0, 12, 4, 12, 99, 1], dtype=np.int32)

day_5_program3 = np.array([3, 21, 1008, 21, 8, 20, 1005, 20, 22, 107, 8, 21, 20, 1006, 20, 31, 1106, 0, 36, 98, 0, 0,
                          1002, 21, 125, 20, 4, 20, 1105, 1, 46, 104, 999, 1105, 1, 46, 1101, 1000, 1, 20, 4, 20,
                          1105, 1, 46, 98, 99], dtype=np.int32)


day_5_test_data = \
    [
        (np.array([3, 9, 8, 9, 10, 9, 4, 9, 99, -1, 8], dtype=np.int32), 8, 1),  # test equal
        (np.array([3, 9, 8, 9, 10, 9, 4, 9, 99, -1, 8], dtype=np.int32), 7, 0),
        (np.array([3, 9, 8, 9, 10, 9, 4, 9, 99, -1, 8], dtype=np.int32), 9, 0),
        (np.array([3, 9, 7, 9, 10, 9, 4, 9, 99, -1, 8], dtype=np.int32), 8, 0),  # test less
        (np.array([3, 9, 7, 9, 10, 9, 4, 9, 99, -1, 8], dtype=np.int32), 7, 1),
        (np.array([3, 9, 7, 9, 10, 9, 4, 9, 99, -1, 8], dtype=np.int32), 9, 0),
        (np.array([3, 3, 1107, -1, 8, 3, 4, 3, 99], dtype=np.int32), 8, 0),
        (np.array([3, 3, 1107, -1, 8, 3, 4, 3, 99], dtype=np.int32), 7, 1),
        (np.array([3, 3, 1107, -1, 8, 3, 4, 3, 99], dtype=np.int32), 9, 0),
        (np.array([3, 3, 1108, -1, 8, 3, 4, 3, 99], dtype=np.int32), 8, 1),
        (np.array([3, 3, 1108, -1, 8, 3, 4, 3, 99], dtype=np.int32), 7, 0),
        (np.array([3, 3, 1108, -1, 8, 3, 4, 3, 99], dtype=np.int32), 9, 0),
        (day_5_program1.copy(), 0, 0),
        (day_5_program1.copy(), 1, 1),
        (day_5_program1.copy(), 20, 1),
        (day_5_program2.copy(), 0, 0),
        (day_5_program2.copy(), 1, 1),
        (day_5_program2.copy(), 20, 1),
        (day_5_program3.copy(), 7, 999),
        (day_5_program3.copy(), 8, 1000),
        (day_5_program3.copy(), 9, 1001),
        (day_5_program3.copy(), 10, 1001)
    ]


day_7_test_data_seq = \
    [
        (np.array([3, 15, 3, 16, 1002, 16, 10, 16, 1, 16, 15, 15, 4, 15, 99, 0, 0], dtype=np.int32),
         [4, 3, 2, 1, 0], 43210),
        (np.array([3, 23, 3, 24, 1002, 24, 10, 24, 1002, 23, -1, 23, 101, 5, 23, 23, 1, 24, 23, 23, 4, 23, 99, 0, 0],
                  dtype=np.int32),
         [0, 1, 2, 3, 4], 54321),
        (np.array([3, 31, 3, 32, 1002, 32, 10, 32, 1001, 31, -2, 31, 1007, 31, 0, 33, 1002, 33, 7, 33, 1, 33, 31, 31,
                   1, 32, 31, 31, 4, 31, 99, 0, 0, 0], dtype=np.int32),
         [1, 0, 4, 3, 2], 65210),
    ]

day_7_test_data_cont = \
    [
        (np.array([3, 26, 1001, 26, -4, 26, 3, 27, 1002, 27, 2, 27, 1, 27, 26, 27, 4, 27, 1001, 28, -1, 28, 1005,
                   28, 6, 99, 0, 0, 5],
                  dtype=np.int32),
         [9, 8, 7, 6, 5], 139629729),
        (np.array([3, 52, 1001, 52, -5, 52, 3, 53, 1, 52, 56, 54, 1007, 54, 5, 55, 1005, 55, 26, 1001, 54,
                   -5, 54, 1105, 1, 12, 1, 53, 54, 53, 1008, 54, 0, 55, 1001, 55, 1, 55, 2, 53, 55, 53, 4,
                   53, 1001, 56, -1, 56, 1005, 56, 6, 99, 0, 0, 0, 0, 10], dtype=np.int32),
         [9, 7, 8, 5, 6], 18216),
    ]


@pytest.mark.parametrize("program,expected", day_2_test_data)
def test_day2(program, expected):
    """Basic instruction processing"""
    computer = IntComputer(program)
    computer.run()
    np.testing.assert_array_equal(expected, computer.memory)


@pytest.mark.parametrize("program,inputs,expected_output", day_5_test_data)
def test_day5(program, inputs, expected_output):
    """Test input / output processing"""
    computer = IntComputer(program)
    computer.input_stream.append(inputs)
    computer.run(False)
    output = computer.output_stream.popleft()
    assert output == expected_output


@pytest.mark.parametrize("program,inputs,expected_output", day_7_test_data_seq)
def test_day7_seq(program, inputs, expected_output):
    """Test sequential linking of computers"""
    computers = [IntComputer(program) for _ in range(len(inputs))]
    for idx, cpu in enumerate(computers[1:]):
        cpu.connect_input(computers[idx].output_stream)
    for idx, setting in enumerate(inputs):
        computers[idx].input_stream.append(setting)
    computers[0].input_stream.append(0)
    for cpu in computers:
        cpu.run(False)
    output = computers[-1].output_stream.popleft()
    assert output == expected_output


@pytest.mark.parametrize("program,inputs,expected_output", day_7_test_data_cont)
def test_day7_cont(program, inputs, expected_output):
    """Test continuous execution of computers in feedback loop"""
    computers = [IntComputer(program) for _ in range(len(inputs))]
    for idx, cpu in enumerate(computers[1:]):
        cpu.connect_input(computers[idx].output_stream)
    computers[0].connect_input(computers[-1].output_stream)
    for idx, setting in enumerate(inputs):
        computers[idx].input_stream.append(setting)
    computers[0].input_stream.append(0)
    halted = [cpu.is_halted() for cpu in computers]
    while not np.all(halted):
        for cpu in computers:
            cpu.run(True)
        halted = [cpu.is_halted() for cpu in computers]

    output = computers[-1].output_stream.popleft()
    assert output == expected_output
