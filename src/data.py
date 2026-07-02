"""segmenting CWRU bearing vibration data."""
import numpy as np
from scipy.io import loadmat


def load_de_signal(path):
    mat = loadmat(path)
    de_key = [k for k in mat.keys() if 'DE_time' in k][0]
    return mat[de_key].flatten()


def segment(signal, window=2048, step=1024):
    return np.array([
        signal[i: i + window]
        for i in range(0, len(signal) - window, step)
    ])