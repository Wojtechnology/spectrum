import math

import matplotlib.pyplot as plt
import numpy as np


def show_spectrogram(
    data: np.array,
    size: int=4096,
    sample_rate: int=44100,
    buffer_size: int=1024,
    stutter_size: int=128,
) -> None:
    num_axes = math.ceil(data.shape[1] / size)
    x_mult = stutter_size / sample_rate;
    y_mult = sample_rate / buffer_size * 2;

    fig, axs = plt.subplots(num_axes, figsize=(20, 10 * num_axes))
    for i, ax in enumerate(axs):
        psm = ax.pcolormesh(data[:, i*size:(i+1)*size])
        fig.colorbar(psm, ax=ax)
        xticks = ax.get_xticks()
        ax.set_xticklabels(['%.2f' % (x_mult * (i * size + val)) for val in xticks])
        yticks = ax.get_yticks()
        ax.set_yticklabels(['%.2f' % (y_mult * val) for val in yticks])

    plt.show()
