.. module:: imcombinepy

.. _fitscombine:

#################
The FITS combiner
#################

This ``fitscombine`` function is made to replace IRAF's `IMCOMBINE
<https://iraf.net/irafhelp.php?val=imcombine&help=Help+Page>`_.

Therefore, it is using the `ndcombine`, but in a way that is designed for FITS file formats. I tried to follow IRAF's IMCOMBINE, while some parts are changed such as parameter names (``lsigma`` / ``hsigma`` to ``sigma`` and/or ``sigma_lower`` / ``sigma_upper``), improved algorithms (sigma-clipped zero and scaling), and not-implemented (``grow`` or ``sigscale``).


********************
Comparison with IRAF
********************
While testing, I found some mysteries of IRAF. For the generation of all test FITS files, I fixed the following arguments to IMCOMBINE::

    IRAF IMCOMBINE        imcombinepy
  * nkeep = 0             nkeep=0, maxrej=ncombine
  * lsigma, hsigma = 2    sigma=(2,2)
  * sigscale = 0          non-zero sigscale not implemented
  * mclip+                cenfunc='median'

i.e., I did **median-centered 2-sigma clipping if rejection is turned on**. The output file name has the following convention::

  <offsets>_<combine>_<reject>_<zero>_<scale>[_mask/_nrej/_sigma].fits

If ``'none'`` is used, it is denoted as ``x``.


Mask inconsistency
------------------
In IRAF IMCOMBINE, we can specify output files. I used, e.g., the following CL sclipt::

  !rm none_med_sc_x_x*.fits
  imcomb *2005UD* combine=med offsets='none' scale='none' zero='none' reject=sigclip lsigma=2 hsigma=2 nkeep=0 sigscale=0 output=none_med_sc_x_x.fits rejmask=mask nrejmasks=nrej sigma=sigma
  imcopy mask.pl none_med_sc_x_x_mask.fits
  imcopy nrej.pl none_med_sc_x_x_nrej.fits
  imcopy sigma.fits none_med_sc_x_x_sigma.fits
  !rm *.pl sigma.fits

The log looks like this::

  Jul 15 10:25: IMCOMBINE
    combine = median, scale = none, zero = none, weight = none
    reject = sigclip, mclip = yes, nkeep = 0
    lsigma = 2., hsigma = 2.
    blank = 0.

                  Images
    bdfc_2005UD_20181012-140207_R_60.0.fits
    bdfc_2005UD_20181012-140703_R_60.0.fits
    bdfc_2005UD_20181012-141159_R_60.0.fits
    bdfc_2005UD_20181012-141654_R_60.0.fits
    bdfc_2005UD_20181012-142150_R_60.0.fits
    bdfc_2005UD_20181012-142646_R_60.0.fits
    bdfc_2005UD_20181012-143142_R_60.0.fits
    bdfc_2005UD_20181012-143637_R_60.0.fits
    bdfc_2005UD_20181012-144133_R_60.0.fits
    bdfc_2005UD_20181012-144629_R_60.0.fits
    bdfc_2005UD_20181012-145126_R_60.0.fits
    bdfc_2005UD_20181012-145621_R_60.0.fits

    Output image = none_med_sc_x_x.fits, ncombine = 12
    Rejection mask = mask.pl
    Number rejected mask = nrej.pl
    Sigma image = sigma
  mask.pl -> none_med_sc_x_x_mask.fits
  nrej.pl -> none_med_sc_x_x_nrej.fits
  sigma.fits -> none_med_sc_x_x_sigma.fits

Here I obtain the 3-D mask ``blahblah_mask.fits``. Then I can check

#. When ``reject=none`` in IRAF,
    - ``mask`` must be all ``False``. (YES)
    - The output must be identical to the naive combination (``np.nanmedian(allimage, axis=0)``) (YES)
#. ``np.sum(mask, axis=0)`` identical to ``nrej``? (YES)
#. Sigma of ``original[~mask]`` along ``axis=0`` identical to output sigma of IRAF? (**NO**)
#. ``((data3d < (comb_iraf - 2*sigma_iraf)) | (data3d > (comb_iraf + 2*sigma_iraf)))`` identical to ``mask``? (**NO**)
#. d

The mystery for me is that the last few checks failed.

.. code-block:: python
  :linenos:

  import glob
  from astropy.stats import sigma_clipped_stats
  from astropy.io import fits
  from matplotlib import pyplot as plt

  fpattern = "testcombine/*2005UD*.fits"
  fpaths = list(glob.glob(fpattern))
  fpaths.sort()
  data3d = []
  for fpath in fpaths:
      data3d.append(fits.open(fpath)[0].data)
  data3d = np.array(data3d)

  res_aspy = sigma_clip(data3d, axis=0, sigma=2, maxiters=1, cenfunc='median',   stdfunc='std', return_bounds=True)
  mask_aspy = res_aspy[0].mask

  fprefix = "none_med_sc_x_x"
  comb_iraf = fits.open(f"testcombine/{fprefix}.fits")[0].data
  mask_iraf = fits.open(f"testcombine/{fprefix}_mask.fits")[0].data.astype(bool)
  sigma_iraf = fits.open(f"testcombine/{fprefix}_sigma.fits")[0].data
  nrej_iraf = fits.open(f"testcombine/{fprefix}_nrej.fits")[0].data

  # Test of item 1&2
  np.testing.assert_array_almost_equal(np.sum(mask_iraf, axis=0), nrej_iraf)
  if fprefix.endswith('_x_x_x'):  # if no rejection happened
      assert np.count_nonzero(mask_iraf) == 0
      assert np.count_nonzero(bn.median(data3d, axis=0) != comb_iraf) == 0

  data3d_nan = data3d.copy()
  data3d_nan[mask_iraf] = np.nan
  comb_iraf_test = bn.nanmedian(data3d_nan, axis=0)
  np.testing.assert_array_almost_equal(comb_iraf_test, comb_iraf)

#. From IRAF's results,
#. IRAF does **not** do the zeroing when both zero and scale are specified.

**************
Documentations
**************
.. autofunction:: imcombinepy.combine.fitscombine


