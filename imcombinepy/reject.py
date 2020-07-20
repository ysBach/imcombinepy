import bottleneck as bn
from astropy.stats import sigma_clip
import numpy as np

from .util import (_set_cenfunc, _set_keeprej, _set_mask, _set_minmax,
                   _set_sigma, _setup_reject)

__all__ = ["sigclip_mask"]


# TODO: let ``cenfunc`` be function object...?
# *************************************************************************** #
# *                               SIGMA-CLIPPING                            * #
# *************************************************************************** #
def _sigclip(
        arr, mask=None, sigma_lower=3., sigma_upper=3., maxiters=5, ddof=0,
        nkeep=3, maxrej=1-1.e-12, cenfunc='median', satlevel=65535,
        irafmode=False
):
    """ Do not provide arr.shape[0] > 65535...
    This is not what one would expect with this kind of simple code.
    """
    # General setup
    _arr, _masks, keeprej, cenfunc, _nvals, lowupp = _setup_reject(
        arr=arr, mask=mask, nkeep=nkeep, maxrej=maxrej, cenfunc=cenfunc
    )
    mask_orig, mask_nkeep, mask_maxrej, mask_pix = _masks
    nkeep, maxrej = keeprej
    nit, ncombine, n_orig = _nvals
    low, upp, low_new, upp_new = lowupp

    # if irafmode:
    #     _arr = np.ma.array(data=_arr, mask=mask_orig | mask_pix)
    #     sc = sigma_clip(_arr, simga_lower=sigma_lower,
    #                     sigma_upper=sigma_upper, cenfunc=cenfunc,
    #                     maxiters=maxiters)
    #     n_reject = np.sum(sc.mask, axis=0)
    #     n_remain = arr.shape[0] - n_reject
    #     mask_nkeep = n_remain < nkeep
    #     mask_maxrej = n_reject > maxrej
    #     # pixels that has many non-NaN values, but too many pixels are
    #     # rejected:
    #     mask_restore = (~mask_pix) & (mask_nkeep | mask_maxrej)
    #     _arr[mask_restore] = cenfunc()

    #     for pix in ~mask_pix and (n_remain < nkeep or n_orig - n_remain > maxrej)

    nrej = ncombine - n_orig  # same as nrej_old at the moment
    n_old = 1*n_orig
    k = 0
    while k < maxiters:
        cen = cenfunc(_arr, axis=0)
        std = bn.nanstd(_arr, axis=0, ddof=ddof)
        low_new[~mask_pix] = (cen - sigma_lower*std)[~mask_pix]
        upp_new[~mask_pix] = (cen + sigma_upper*std)[~mask_pix]

        # In numpy, > or < automatically applies along axis=0!!
        mask_bound = (_arr < low_new) | (_arr > upp_new)
        _arr[mask_bound] = np.nan

        n_new = ncombine - np.sum(mask_bound, axis=0)
        n_change = n_old - n_new
        total_change = np.sum(n_change)

        mask_nochange = (n_change == 0)  # identical to say "max-iter reached"
        mask_nkeep = ((ncombine - nrej) < nkeep)
        mask_maxrej = (nrej > maxrej)

        # mask pixel position if any of these happened.
        # Including mask_nochange here will not change results but only
        # spend more time.
        mask_pix = mask_nkeep | mask_maxrej

        # revert to the previous ones if masked.
        # By doing this, pixels which was mask_nkeep now, e.g., will
        # again be True in mask_nkeep in the next iter but unchanged.
        # This should be done at every iteration (unfortunately)
        # because, e.g., if nkeep is very large, excessive rejection may
        # happen for many times, and the restoration CANNOT be done
        # after all the iterations.
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

    mask = mask_orig | (arr < low_new) | (arr > upp_new)

    code = np.zeros(_arr.shape[1:], dtype=np.uint8)
    if (maxiters == 0):
        code += 1
    else:
        code += (2*mask_nochange + 4*mask_nkeep
                 + 8*mask_maxrej).astype(np.uint8)

    if irafmode:
        n_minimum = max(nkeep, ncombine - maxrej)
        resid = np.abs(_arr - cen)
        # need this cuz bn.argpartition cannot handle NaN:
        resid[mask_orig] = satlevel  # very large value

        ind = bn.argpartition(resid, kth=n_minimum, axis=0)
        # need *nan*max because some positions can have many NaNs.
        resid_cut = bn.nanmax(np.take_along_axis(resid, ind, axis=0), axis=0)
        # Note that ``np.nan <= np.nan`` is ``False``, so NaN pixels are
        # not affected by this:
        mask[resid <= resid_cut] = False

    return (mask, low, upp, nit, code)


# == high-level ============================================================= #
def sigclip_mask(
    arr, mask=None, sigma=3., sigma_lower=None, sigma_upper=None, maxiters=5,
    ddof=0, nkeep=3, maxrej=None, cenfunc='median',
    axis=0, full=True, satlevel=65535, irafmode=False
):
    ''' Only along axis=0.

    Parameters
    ----------
    arr : ndarray
        The array to be subjected for masking. ``arr`` and ``mask`` must
        have the identical shape.
    mask : ndarray, optional.
        The initial mask provided prior to any rejection. ``arr`` and
        ``mask`` must have the identical shape.
    sigma : float-like, optional.
        The sigma-factors to be used for sigma-clip rejeciton.
        Overridden by ``sigma_lower`` and/or ``sigma_upper``, if input.
        See Note.
    sigma_lower : float or `None`, optional
        The number of standard deviations to use as the lower bound for
        the clipping limit.  If `None` then the value of ``sigma`` is
        used.  The default is `None`.
    sigma_upper : float or `None`, optional
        The number of standard deviations to use as the upper bound for
        the clipping limit.  If `None` then the value of ``sigma`` is
        used.  The default is `None`.
    maxiters : int, optional.
        The maximum number of iterations to do the sigma-clipping. It is
        silently converted to int if it is not.
    ddof : int, optional.
        The delta-degrees of freedom (see `numpy.std`). It is silently
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

    Returns
    -------
    o_mask : ndarray of bool
        The mask of the same shape as ``arr`` and ``mask``. Note the
        oritinal ``mask`` is propagated, so pure sigma-clipping mask is
        obtained by ``o_mask^mask``, because all pixel
    o_low, o_upp : ndarray of ``dtype=dtype``
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
        Returned only if ``full = True``. Each element is a ``uint8``
        value with::
          *      (0): maxiters reached without any flag below
          * 1-th (1): maxiters == 0 (no iteration happened)
          * 2-th (2): iteration finished before maxiters reached
          * 3-th (4): remaining ndata < nkeep reached
          * 4-th (8): rejected ndata > maxrej reached
        The code of 10 is, for example, 1010 in binary, so the iteration
        finished before ``maxiters`` (2-th flag) because pixels more
        than ``maxrej`` are rejected (4-th flag).

    Notes
    -----
    The central value is first determined by ``cenfunc``. The "sigma"
    value is calculated by ``nanstd`` with the given ``ddof``. After the
    first iteration, any value ``arr < sigma_lower*std`` or
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

    See `bench_isnan.md`_ why ``nanXXX`` functions are used.

    .. _bench_isnan.md: https://github.com/ysBach/imcombinepy/tree/master/bench/bench_isnan.md

    .. note::
        The number of rejected points: ``np.sum(o_mask, axis=0)``.

        The mask excluding original mask, i.e., only the points that are
        rejected by rejection is ``o_mask ^ mask`` (because ``o_mask =
        mask | rejected_positions``).
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
        satlevel=satlevel,
        irafmode=irafmode
    )
    if full:
        return o_mask, o_low, o_upp, o_nit, o_code
    else:
        return o_mask


# *************************************************************************** #
# *                             MINMAX CLIPPING                             * #
# *************************************************************************** #
def _minmax(
        arr, mask=None, q_low=0, q_upp=0, cenfunc='median'
):
    # General setup (nkeep and maxrej as dummy)
    _arr, _masks, _, cenfunc, _nvals, lowupp = _setup_reject(
        arr=arr, mask_orig=mask, nkeep=None, maxrej=None, cenfunc=cenfunc
    )
    mask = _masks[0]  # nkeep and maxrej are not used in MINMAX.
    nit, n_orig, n_old = _nvals
    low, upp, low_new, upp_new = lowupp
    nrej = n_orig - n_old  # same as nrej_old at the moment

    n_rej_low = int(n_old * q_low + 0.001)  # adding 0.001 following IRAF
    n_rej_upp = int(n_old * q_upp + 0.001)  # adding 0.001 following IRAF

    while True:
        arr[arr <= low] = np.nan
        n_after_low = n_orig - np.isnan(arr, axis=0)

        minval = bn.nanargmin(_arr, axis=0)


# *************************************************************************** #
# *                       PERCENTILE CLIPPING (PCLIP)                       * #
# *************************************************************************** #

# *************************************************************************** #
# *                    CCD NOISE MODEL CLIPPING (CCDCLIP)                   * #
# *************************************************************************** #
