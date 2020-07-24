# imcombinepy

A python package to replace IRAF imcombine with both python and CLI interface using bottleneck.

Simplest use case:

```python
import imcombinepy as imc

fpattern="/path/to/image/directory/SNUO*.fits"

kw = dict(combine='med', scale="median", reject='sc', sigma=(2, 2), memlimit=4.e+9)

comb_wcs = imc.fitscombine(fpattern=fpattern, offsets="wcs", **kw)
comb_img = imc.fitscombine(fpattern=fpattern, offsets=None, **kw)
```

It selects all `SNUO*.fits` files, and offsets using WCS (`comb_wcs`) or naive combination (`comb_img`). Internally it does 2-sigma clipping (`reject` and `sigma`) for upper/lower, centering for sigma clipping is median (default of `cenfunc`). Each image will be scaled by a factor such that `scale[i]` for the `i`-th image is `median(image[i]) / median(image[0])`, following IRAF.

One can also use ``pathlib``:
```python
from pathlib import Path
TOPPATH = Path('.')
allfits = list(TOPPATH.glob("*.fits"))
allfits.sort()

comb_wcs = imc.fitscombine(fpaths=allfits, offsets="wcs", **kw)
comb_img = imc.fitscombine(fpaths=allfits, offsets=None, **kw)
```

You may play more:

```python
res = imc.fitscombine(
    fpattern=fpattern,
    offsets="wcs",
    combine='med',
    scale="median_sc",
    scale_kw=dict()
    zero="avg",
    reject='ccd',
    sigma=(2, 2),
    verbose=True,
    full=True,
    nkeep=3,
    maxrej=5,
    output="test.fits",
    output_nrej="test_nrej.fits",
    output_mask="test_mask.fits",
    output_low="test_low.fits",
    output_upp="test_upp.fits",
    output_sigma="test_sigma.fits",
    output_rejcode="test_rejcode.fits",
    overwrite=True,
    memlimit=4.e+9
)
comb, sigma, mask_total, mask_rej, mask_thresh, low, upp, nit, rejcode = res
```


# Limitations (Future Works)
1. Chunked combine using memlimit is not available yet.
1. CLI is not supported yet.
