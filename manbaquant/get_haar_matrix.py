import torch
import numpy as np
import pywt

def get_wavelet_matrix(wavelet_name, size):
    """
    生成给定小波的变换矩阵。

    参数:
    wavelet_name (str): 小波的名称，例如 'haar', 'db1', 'coif1' 等。
    size (int): 矩阵的大小（通常为 2 的幂次）。

    返回:
    torch.Tensor: 小波变换矩阵。
    """
    wavelet = pywt.Wavelet(wavelet_name)
    matrix = np.zeros((size, size))
    
    for i in range(size):
        impulse = np.zeros(size)
        impulse[i] = 1
        coeffs = pywt.wavedec(impulse, wavelet, mode='per', level=int(np.log2(size)))
        matrix[:, i] = pywt.waverec(coeffs, wavelet, mode='per')
    
    return torch.tensor(matrix, dtype=torch.float32)