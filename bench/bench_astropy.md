```python
import numpy as np
import bottleneck as bn
from astropy.stats import SigmaClip, sigma_clip
from imcombinepy.reject import sigclip_mask


np.random.seed(12345)
data3d = np.random.normal(size=(10, 1000, 2000))
mask3d = np.zeros_like(data3d).astype(bool)

sckw = dict(sigma=2, maxiters=1)
askw = dict(**sckw, cenfunc=np.nanmedian, stdfunc=np.nanstd, return_bounds=True)
bnkw = dict(**sckw, cenfunc='median', stdfunc='std', return_bounds=True)

res3d = sigclip_mask(data3d, mask3d, **sckw, maxrej=data3d.size)
resas3d = sigma_clip(data3d, **bnkw, axis=0)
np.testing.assert_array_almost_equal(res3d[0], resas3d[0].mask)

print("="*80)
print("All tests passed. Timing benchmark starts")
print("="*80)

print(f"data.shape = {data3d.shape}")
print("    astropy + numpy     ", end=" : ")
%timeit -n 1 -r 5 sigma_clip(data3d, **askw, axis=0)
print("    astropy + bottleneck", end=" : ")
%timeit -n 1 -r 5 sigma_clip(data3d, **bnkw, axis=0)
print("      This       pkg    ", end=" : ")
%timeit -n 1 -r 5 sigclip_mask(data3d, mask3d, **sckw, maxrej=data3d.size)
print()


data3d_nan = data3d.copy()
mask3d_nan = mask3d.copy()
mask3d_nan[3, 10:20, 10:25] = True
mask3d_nan[4, 89:99, 60:70] = True
data3d_nan[mask3d_nan] = np.nan

res3d_nan = sigclip_mask(data3d_nan, mask=mask3d_nan, **sckw, maxrej=data3d.size)
resas3d_nan = sigma_clip(data3d_nan, **bnkw, axis=0)
np.testing.assert_array_almost_equal(res3d_nan[0] | np.isnan(data3d_nan),
resas3d_nan[0].mask)

print(f"data.shape = {data3d.shape} with nan values")
print("    astropy + numpy     ", end=" : ")
%timeit -n 1 -r 5 sigma_clip(data3d_nan, **askw, axis=0)
print("    astropy + bottleneck", end=" : ")
%timeit -n 1 -r 5 sigma_clip(data3d_nan, **bnkw, axis=0)
print("      This       pkg    ", end=" : ")
%timeit -n 1 -r 5 sigclip_mask(data3d_nan, mask3d_nan, **sckw, maxrej=data3d.size)


# ===============================================================================
# All tests passed. Timing benchmark starts
# ================================================================================
# data.shape = (10, 1000, 2000)
#     astropy + numpy      : 1.68 s +/- 53.9 ms per loop
#     astropy + bottleneck : 575 ms +/- 21 ms per loop
#       This       pkg     : 750 ms +/- 10.3 ms per loop

# data.shape = (10, 1000, 2000) with nan values
#     astropy + numpy      : 1.73 s +/- 45.6 ms per loop
#     astropy + bottleneck : 599 ms +/- 2.08 ms per loop
#       This       pkg     : 749 ms +/- 3.37 ms per loop
```

The time scales almost linearly for bottleneck, while it is slightly steeper in astropy. Testing with ``(10, 1000, 200)`` shape:
```
================================================================================
All tests passed. Timing benchmark starts
================================================================================
data.shape = (10, 1000, 200)
    astropy + numpy      : 138 ms +/- 6.93 ms per loop
    astropy + bottleneck : 51 ms +/- 2.11 ms per loop
      This       pkg     : 72.3 ms +/- 929 µs per loop

data.shape = (10, 1000, 200) with nan values
    astropy + numpy      : 134 ms +/- 4.06 ms per loop
    astropy + bottleneck : 48.9 ms +/- 625 µs per loop
      This       pkg     : 69.8 ms +/- 896 µs per loop
```