import numpy as np


def trans_pred(predicciones):
    """
    Transforma las predicciones negativas en cero.

    """
    return np.maximum(predicciones, 0)
