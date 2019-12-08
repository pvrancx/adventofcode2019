import numpy as np


def readfile(inputfile: str) -> str:
    with open(inputfile) as file:
        return file.read()[:-1]  # remove newline


def str_2_np(txt: str) -> np.ndarray:
    return np.array([int(d) for d in list(txt)])


def get_image(txt: str, width: int, height: int) -> np.ndarray:
    data = str_2_np(txt)
    assert data.size % (width * height) == 0, 'Invalid dimensions'
    n_layers = data.size // (width * height)
    return data.reshape((n_layers, height, width))


def checksum(image: np.ndarray) -> int:
    n_zeros = np.sum(image == 0, axis=(1, 2))
    min_layer = np.argmin(n_zeros)
    return np.sum(image[min_layer] == 1) * np.sum(image[min_layer] == 2)


def visible_image(image: np.ndarray) -> np.ndarray:
    result = np.ones_like(image[-1], dtype=int) * 2
    for layer in image:
        idx = (result == 2)
        result[idx] = layer[idx]
    return result


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    def _main():
        image = get_image(readfile('../inputs/day8.txt'), 25, 6)
        print(checksum(image))
        plt.matshow(visible_image(image))
        plt.show()
    _main()
