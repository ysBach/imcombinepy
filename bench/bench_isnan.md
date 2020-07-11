From the benchmark below, I find
1. The ``nanXXX`` function is of course slower than ``XXX`` function.
1. The ``median`` and ``nanmedian`` of bottleneck has nearly no speed difference (!)
1. Testing the existence of at least one NaN has a little bit of overhead.


```python
import numpy as np
import bottleneck as bn

np.random.seed(12345)
data3d = np.random.normal(size=(10, 1000, 200))
mask3d = np.zeros_like(data3d).astype(bool)
test3d = data3d.copy()
test3d[1, 5, 10] = np.nan
test3d[9, 200, 100] = np.nan
tmsk3d = mask3d.copy()
tmsk3d[3, 10, 10] = True
tmsk3d[4, 89, 60] = True

print("When data contains no NaN")
print("np.mean(data_nonan, axis=0)      :", end=' ')
%timeit -n 1 -r 5 np.mean(data3d, axis=0)
print()
print("np.median(data_nonan, axis=0)    :", end=' ')
%timeit -n 1 -r 5 np.median(data3d, axis=0)
print("bn.median(data_nonan, axis=0)    :", end=' ')
%timeit -n 1 -r 5 bn.median(data3d, axis=0)

print()
print("np.nanmean(data_nonan, axis=0)   :", end=' ')
%timeit -n 1 -r 5 np.nanmean(data3d, axis=0)
print("bn.nanmean(data_nonan, axis=0)   :", end=' ')
%timeit -n 1 -r 5 bn.nanmean(data3d, axis=0)

print()
print("np.nanmedian(data_nonan, axis=0) :", end=' ')
%timeit -n 1 -r 5 np.nanmedian(data3d, axis=0)
print("bn.nanmedian(data_nonan, axis=0) :", end=' ')
%timeit -n 1 -r 5 bn.nanmedian(data3d, axis=0)

print()
print("Now data contains NaN")
print("np.nanmean(data_nan, axis=0)     :", end=' ')
%timeit -n 1 -r 5 np.nanmean(test3d, axis=0)
print("bn.nanmean(data_nan, axis=0)     :", end=' ')
%timeit -n 1 -r 5 bn.nanmean(test3d, axis=0)
print()

print("np.nanmedian(data_nan, axis=0)   :", end=' ')
%timeit -n 1 -r 5 np.nanmedian(test3d, axis=0)
print("bn.nanmedian(data_nan, axis=0)   :", end=' ')
%timeit -n 1 -r 5 bn.nanmedian(test3d, axis=0)

print()
%timeit -n 1 -r 5 np.any(np.isnan(test3d))

# When data contains no NaN
# np.mean(data_nonan, axis=0)      : 1.49 ms +/- 106 µs per loop

# np.median(data_nonan, axis=0)    : 40.6 ms +/- 2.11 ms per loop
# bn.median(data_nonan, axis=0)    : 20.8 ms +/- 1.15 ms per loop

# np.nanmean(data_nonan, axis=0)   : 9.54 ms +/- 798 µs per loop
# bn.nanmean(data_nonan, axis=0)   : 3.31 ms +/- 81.7 µs per loop

# np.nanmedian(data_nonan, axis=0) : 108 ms +/- 2.92 ms per loop
# bn.nanmedian(data_nonan, axis=0) : 21.6 ms +/- 1.93 ms per loop

# Now data contains NaN
# np.nanmean(data_nan, axis=0)     : 9.51 ms +/- 688 µs per loop
# bn.nanmean(data_nan, axis=0)     : 3.43 ms +/- 238 µs per loop

# np.nanmedian(data_nan, axis=0)   : 101 ms +/- 942 µs per loop
# bn.nanmedian(data_nan, axis=0)   : 21.3 ms +/- 2.04 ms per loop

# 1.24 ms +/- 128 µs per loop

```

By tuning the size of ``data3d`` to ``(10, 1000, 2000)`` (10 times larger data), I found
1. Most numpy functions take longer time than that expectred from a linear extrapolation (x10+ slower)
1. All bottleneck functions scale nearly linearly (better than numpy for long computation)
1. NaN checking scales quite linearly.

Therefore, the conclusion is
* Question: Is it better to follow this algorithm: "check if there's any nan in data, if no, use ``np.mean`` to boost the calculation".
* Answer: For our case, **NO**. If ``bn.nanmean`` takes time ``t_bn``, ``np.mean`` takes time ``t_np``, and ``np.any(np.isnan(data))`` takes ``t_nan``, roughly ``t_bn ~ t_np + t_nan``, so there's only little gain (sometimes even slower).

Therefore, I did not use, e.g., ``np.mean``, but always used ``bn.nanmean``.
