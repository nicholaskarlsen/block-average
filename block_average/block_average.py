import numpy as np
from numba import jit


def block_average(data, block_size=None):
    """ Computes the block average of a dataset for a single, or multiple block sizes depening on the input.
    If no block size is given, the block averaging will be performed for all block sizes from 1 to len(data)/4, in 
    this case, an array containting the block sizes will also be returned.

    Args:
        data: Array containting data
        block_size:
    Returns:
    """

    if isinstance(block_size, int):
        return single_block_size(data, block_size)

    elif isinstance(block_size, np.ndarray):
        return multiple_block_sizes(data, block_size)

    elif isinstance(block_size, list):
        # convert to ndarray to avoid problems with njit and amax
        return multiple_block_sizes(data, np.array(block_size))

    elif block_size is None:
        # need at least 4 blocks to compute variance
        block_size = np.arange(1, len(data) // 4)
        return multiple_block_sizes(data, block_size)

    else:
        raise TypeError("%s not valid type for block_size. Expected: int, list or none" % type(block_size))

    return


@jit(nopython=True)
def single_block_size(data, block_size: int):
    """ Block-averaging for a single block size
    Args:
        data (np.ndarray): Array containting data
        block_size (int): Block size
    Returns:
        block_avg (float), block_var (float)
    """
    steps = len(data)
    num_blocks = steps // block_size
    blocks = np.empty(num_blocks)

    for i in range(num_blocks):
        a = i * block_size
        b = a + block_size
        blocks[i] = np.mean(data[a:b])

    block_avg = np.mean(blocks)
    block_var = np.var(blocks) / (num_blocks - 1)
    return block_avg, block_var


@jit(nopython=True)
def multiple_block_sizes(data, block_size):
    """ Block averaging for multiple block size
    Args:
        data (np.ndarray): Array containting data
        block_size (np.ndarray): Array containting block_sizes
    Returns:
        block_avg (np.ndarray), block_var (np.ndarray), block_size (np.ndarray)
    """
    steps = len(data)
    max_block_size = np.max(block_size)

    if max_block_size > int(steps // 4):
        raise AssertionError("Maximum block size less that N / 4, can not compute variance!")

    num_block_sizes = len(block_size)
    block_avg = np.zeros(num_block_sizes)
    block_var = np.zeros(num_block_sizes)

    for i in range(num_block_sizes):
        block_avg[i], block_var[i] = single_block_size(data=data, block_size=block_size[i])
    return block_avg, block_var, block_size
