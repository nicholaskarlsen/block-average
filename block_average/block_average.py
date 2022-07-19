import numpy as np
from numba import jit


def block_average(data, block_size=None):

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
        raise TypeError(
            "%s not valid type for block_size. Expected: int, list or none"
            % type(block_size)
        )

    return


@jit(nopython=True)
def single_block_size(data, block_size: int):
    """
    args:
        data (np.ndarray) :
        block_size (int) :
    returns:
        block_avg , block_var (tuple[float, float])
    """
    steps = len(data)
    num_blocks = steps // block_size  # number of blocks
    blocks = np.empty(num_blocks)

    for i in range(num_blocks):
        a = i * block_size  # start of the block
        b = a + block_size  # end of the block
        blocks[i] = np.mean(data[a:b])  # mean of the block

    # compute mean and variance of the blocks
    block_avg = np.mean(blocks)
    block_var = np.var(blocks) / (num_blocks - 1)

    return block_avg, block_var


@jit(nopython=True)
def multiple_block_sizes(data, block_size):
    steps = len(data)  # total number of observations in x

    max_block_size = np.max(block_size)

    if max_block_size > int(steps // 4):
        raise AssertionError(
            "Maximum block size less that N / 4, can not compute variance!"
        )

    num_block_sizes = len(block_size)  # total number of block sizes
    block_avg = np.zeros(num_block_sizes)  # mean, should be roughly constant
    block_var = np.zeros(num_block_sizes)  # variance for each block size

    for i in range(num_block_sizes):
        block_avg[i], block_var[i] = single_block_size(
            data=data, block_size=block_size[i]
        )

    return block_avg, block_var, block_size
