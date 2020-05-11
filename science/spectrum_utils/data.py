import numpy as np


def load_spectrogram(file: str) -> np.array:
    with open(file) as f:
        o = f.read()

    lines = o.strip().split('\n')
    return np.array([[float(v) for v in l.strip().split(' ')] for l in lines[1:]]).transpose()

