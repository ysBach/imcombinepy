import numpy as np
from .util import _set_keeprej, _set_cenfunc, _set_mask, _set_sigma
import bottleneck as bn


__all__ = ["sigclip_mask"]


'''
Among the 6 rejection algorithms in IRAF,
```
         none - No rejection
      sigclip - Reject pixels using a sigma clipping algorithm
       minmax - Reject the nlow and nhigh pixels
        pclip - Reject pixels using sigma based on percentiles
      ccdclip - Reject pixels using CCD noise parameters
     crreject - Reject only positive pixels using CCD noise parameters
    avsigclip - Reject pixels using an averaged sigma clipping algorithm
```

avsigclip and crreject are not implemented as

    (1) ``avsigclip`` is ccdclip with readout noise is fixed 0.
    (2) ``crreject`` is ccdclip with ``lsigma`` or ``sigma_lower=inf``.

Parameter mappings (<IRAF>: <python>):
    * nkeep: nkeep, maxrej [ALL]
    * mclip: cenfunc (select median or mean) [ccdclip]
    * lsigma, hsigma: sigma_lower, sigma_upper [sigclip, pclip, ccdclip]
    * rdnoise, gain, snoise: rdnoise, gain, snoise [ccdclip]
    * sigscale
    * pclip
    * grow

'''


# TODO: let ``cenfunc`` be function object...?
# *************************************************************************** #
# *                               SIGMA-CLIPPING                            * #
# *************************************************************************** #
def _sigclip(
        arr, mask=None, sigma_lower=3., sigma_upper=3., maxiters=5, ddof=0,
        nkeep=3, maxrej=1-1.e-12, cenfunc='median'
):
    """ Do not provide arr.shape[0] > 65535...
    This is not what one would expect with this kind of simple code.
    """
    mask = _set_mask(arr, mask)
    _arr = arr.copy()
    _arr[mask] = np.nan
    numnan = np.sum(mask, axis=0)
    nkeep, maxrej = _set_keeprej(_arr, nkeep, maxrej, axis=0)
    cenfunc = _set_cenfunc(cenfunc, nameonly=False, nan=True)

    # unsigned 8/16-bit (up to 255/65535) will be enough for num_iter
    # and num_rej, respectively.
    nit = np.ones(_arr.shape[1:], dtype=np.uint8)
    n_orig = _arr.shape[0]
    nrej_old = np.sum(mask, axis=0, dtype=np.uint16)
    n_old = n_orig - nrej_old  # num of remaining pixels

    # Initiate
    low = bn.nanmin(_arr, axis=0)
    upp = bn.nanmax(_arr, axis=0)
    low_new = low.copy()
    upp_new = upp.copy()

    nrej = n_orig - n_old  # same as nrej_old at the moment
    mask_nkeep = ((n_orig - numnan) < nkeep)
    mask_maxrej = (nrej_old > maxrej)
    mask_pix = mask_nkeep | mask_maxrej
    k = 0
    while k < maxiters:
        cen = cenfunc(_arr, axis=0)
        std = bn.nanstd(_arr, axis=0, ddof=ddof)
        low_new[~mask_pix] = (cen - sigma_lower*std)[~mask_pix]
        upp_new[~mask_pix] = (cen + sigma_upper*std)[~mask_pix]

        # In numpy, > or < automatically applies along axis=0!!
        mask_bound = (_arr < low_new) | (_arr > upp_new)
        _arr[mask_bound] = np.nan

        n_new = n_orig - np.sum(mask_bound, axis=0)
        n_change = n_old - n_new
        total_change = np.sum(n_change)
        mask_nochange = (n_change == 0)  # identical to say "max-iter reached"
        mask_nkeep = ((n_orig - nrej) < nkeep)
        mask_maxrej = (nrej > maxrej)

        # mask pixel position if any of these happened.
        # Including mask_nochange here will not change results but only
        # spend more time.
        mask_pix = mask_nkeep | mask_maxrej

        # revert to the previous ones if masked.
        # By doing this, pixels which was mask_nkeep now, e.g., will
        # again be True in mask_nkeep in the next iter but unchanged.
        low_new[mask_pix] = low[mask_pix]
        upp_new[mask_pix] = upp[mask_pix]

        if total_change == 0:
            break

        if np.all(mask_pix):
            break

        # update only non-masked pixels
        nrej[~mask_pix] = n_change[~mask_pix]
        # update only changed pixels
        nit[~mask_nochange] += 1
        k += 1
        n_old = n_new

    mask |= (arr < low_new) | (arr > upp_new)

    code = np.zeros(_arr.shape[1:], dtype=np.uint8)
    code += (2*mask_nochange + 4*mask_nkeep + 8*mask_maxrej).astype(np.uint8)

    if (maxiters == 0):
        code += 1

    return (mask, low, upp, nit, code)


# == high-level ============================================================= #
def sigclip_mask(
    arr, mask=None, sigma=3., sigma_lower=None, sigma_upper=None, maxiters=5,
    ddof=0, nkeep=3, maxrej=None, cenfunc='median',
    axis=0, full=True,
):
    ''' only along axis=0
    Parameters
    ----------
    arr : nd-array
        The array to be subjected for masking. ``arr`` and ``mask`` must
        have the identical shape.
    mask : nd-array, optional.
        The initial mask provided prior to any rejection. ``arr`` and
        ``mask`` must have the identical shape.
    sigma : float-like, optional.
        The sigma-factors to be used for sigma-clip rejeciton. See Note.
    maxiters : int, optional.
        The maximum number of iterations to do the sigma-clipping. It is
        silently converted to int if it is not.
    ddof : int, optional.
        The delta-degrees of freedom (see `~np.std`). It is silently
        converted to int if it is not.
    nkeep : float or int, optional.
        The minimum number of pixels that should be left after
        rejection. If ``nkeep < 1``, it is regarded as fraction of the
        total number of pixels along the axis to combine. See Note.
    maxrej : float or int, optional.
        The maximum number of pixels that can be rejected during the
        rejection. If ``maxrej < 1``, it is regarded as fraction of the
        total number of pixels along the axis to combine. In IRAF,
        negative ``nkeep`` is regarded as ``maxrej`` of this function,
        i.e., only one of minimum number to keep and maxumum number to
        reject can be set in IRAF. See Note.
    cenfunc : str, optional.
        The centering function to be used.
          * median if  ``cenfunc in ['med', 'medi', 'median']``
          * average if ``cenfunc in ['avg', 'average', 'mean']``
          * lower median if ``cenfunc in ['lmed', 'lmd', 'lmedian']``
        The lower median means the median which takes the lower value
        when even number of data is left. This is suggested to be robust
        against cosmic-ray hit according to IRAF IMCOMBINE manual.
    axis : int, optional.
        The axis to combine the image.
    full : bool, optional.
        Whether to return full results. See Return.

    Return
    ------
    o_mask : ndarray of bool
        The mask of the same shape as ``arr`` and ``mask``. Note the
        oritinal ``mask`` is propagated, so pure sigma-clipping mask is
        obtained by ``o_mask^mask``, because all pixel
    o_low, o_upp : ndarray of flat
        Returned only if ``full = True``. The lower and upper bounds
        used for sigma clipping. Data with ``(arr < o_low) | (o_upp <
        arr)`` are masked. Shape of ``arr.shape[1:]``.
    o_nit : ndarray of int or int
        Returned only if ``full = True``. The number of iterations until
        it is halted. If ``simple = False``, shape of ``arr.shape[1:]``,
        i.e., the number for each pixel in ``(n-1)``-D. If ``simple =
        True``, this is an int, the maximum value of the otherwise
        ``(n-1)``-D array.
    o_code : ndarray of uint8
        Returned only if ``full = True``. Each element is a uint8 value
        with
          *      (0): maxiters reached without any flag below
          * 1-th (1): maxiters == 0 (no iteration happened)
          * 2-th (2): iteration finished before maxiters reached
          * 3-th (4): remaining ndata < nkeep reached
          * 4-th (8): rejected ndata > maxrej reached

    Note
    ----
    The central value is then first determined by ``cenfunc``. The
    "sigma" value is calculated by ``nanstd`` with the given ``ddof``.
    After the first iteration, any value ``arr < sigma_lower*std`` or
    ``sigma_upper*std < arr`` is replaced with ``nan`` internally (not
    modifying the original ``arr`` of course). At each position (pixel
    for 2-D image, voxel for 3-D image, etc), the number of rejected
    points are calculated. If this exceeds ``maxrej`` or if the
    remaining non-nan value is fewer than ``nkeep``, the rejection is
    reverted to the previous iteration.

    Sometimes ``maxiters = 0`` is given, and in such case, the lower and
    upper bounds are nothing but ``nanmin`` and ``nanmax`` at the
    position, and number of iteration is 0. It sometimes happens that
    the number of remaining pixels at the position is fewer than
    ``nkeep`` even before any rejection due to the severe masking or
    many nan values at the position. Similarly ``maxrej`` condition may
    be met even before any rejection. Then only ``nanmin`` and
    ``nanmax`` will be given as lower and upper bounds as above. The
    ``o_code`` will hint what happened.

    See ``bench_isnan.md`` why I only used ``nanXXX`` functions.

    Tips are
    1. The number of rejected points: ``np.sum(o_mask, axis=0)``
    2. The mask excluding original mask, i.e., only the points that are
       rejected by rejection is ``o_mask ^ mask`` (because ``o_mask =
       mask | rejected_positions``)
    '''
    if axis != 0:
        raise ValueError("Currently only axis=0 is supported")

    mask = _set_mask(arr, mask)
    sigma_lower, sigma_upper = _set_sigma(sigma, sigma_lower, sigma_upper)
    nkeep, maxrej = _set_keeprej(arr, nkeep, maxrej, axis)
    cenfunc = _set_cenfunc(cenfunc)
    maxiters = int(maxiters)
    ddof = int(ddof)

    o_mask, o_low, o_upp, o_nit, o_code = _sigclip(
        arr=arr,
        mask=mask,
        sigma_lower=sigma_lower,
        sigma_upper=sigma_upper,
        maxiters=maxiters,
        ddof=ddof,
        nkeep=nkeep,
        maxrej=maxrej,
        cenfunc=cenfunc,
    )
    if full:
        return o_mask, o_low, o_upp, o_nit, o_code
    else:
        return o_mask

# *************************************************************************** #
# *                             MINMAX CLIPPING                             * #
# *************************************************************************** #


# *************************************************************************** #
# *                       PERCENTILE CLIPPING (PCLIP)                       * #
# *************************************************************************** #

# *************************************************************************** #
# *                    CCD NOISE MODEL CLIPPING (CCDCLIP)                   * #
# *************************************************************************** #
