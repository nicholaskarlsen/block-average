# block-average
A minimal python package that computes the block average of data.

## How to use

```python
from block_average import block_average

# Compute block average with a block size of 3
block_mean, block_var = block_average(x, 3)

# Compute block average with multiple block sizes
block_mean, block_var, block_size = block_average(x, [1, 2, 3])

# Compute block average for all possible block sizes from 1 to len(x)/4
block_mean, block_var, block_size = block_average(x)
```

## Example Output

![signal](https://github.com/nicholaskarlsen/block-average/blob/main/example/signal.png?raw=true?raw=true)
![std](https://github.com/nicholaskarlsen/block-average/blob/main/example/std.png?raw=true)
