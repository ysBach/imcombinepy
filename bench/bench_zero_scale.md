# Benchmark related to zero and scale calculations

## Finding mean of each frame
The easiest way to find the mean of each frame, to be used for zeroing or scaling, is to use for loop and append the mean of each image to an initially empty array. Another idea is to use ``numpy.apply_over_axes``. As one would expect, the latter is much slower:

```python
np.testing.assert_array_almost_equal(
    np.array([bn.nanmean(data3d[i,]) for i in range(data3d.shape[0])]),
    np.apply_over_axes(bn.nanmean, data3d, axes=[1, 2]).ravel()
)

%timeit -n 1 -r 10 np.array([bn.nanmean(data3d[i,]) for i in range(data3d.shape[0])])
%timeit -n 1 -r 10 np.apply_over_axes(bn.nanmedian, data3d, axes=0)

# 4.34 ms +/- 234 µs per loop (mean +/- std. dev. of 10 runs, 1 loop each)
# 45.8 ms +/- 988 µs per loop (mean +/- std. dev. of 10 runs, 1 loop each)
```
