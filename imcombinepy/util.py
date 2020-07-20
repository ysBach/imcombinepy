import bottleneck as bn
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.time import Time
from .numpy_util import lmedian, nanlmedian

__all__ = [
    'write2fits', 'update_hdr',
    'do_zs', 'get_zsw',
    '_get_combine_shape', '_set_int_dtype',
    '_setup_reject',
    '_set_mask', '_set_sigma', '_set_keeprej', '_set_minmax',
    '_set_thresh_mask',
    '_set_cenfunc', '_set_combfunc',
    '_set_reject_name', '_set_calc_zsw',
    "slice_from_string"
]


def _get_from_hdr(header, key):
    pass


def do_zs(arr, zeros, scales, copy=False):
    if copy:
        arr = arr.copy()
    # below two took less than 10 us for 100 images
    all0 = np.all(zeros == 0) or zeros is None
    all1 = np.all(scales == 1) or scales is None
    if not all0 and not all1:  # both zero and scale
        for i in range(arr.shape[0]):
            arr[i, ] = (arr[i, ] - zeros[i])/scales[i]
    elif not all0 and all1:  # zero only
        for i in range(arr.shape[0]):
            arr[i, ] = arr[i, ] - zeros[i]
    elif all0 and not all1:  # scale only
        for i in range(arr.shape[0]):
            arr[i, ] = arr[i, ]/scales[i]
    # Times:
    #   (np.random.normal(size=(1000,1000)) - 0)/1   21.6 ms +- 730 us
    #   np.random.normal(size=(1000,1000))           20.1 ms +- 197 us

    return arr


def get_zsw(arr, zero, scale, weight, zero_kw, scale_kw, zero_to_0th,
            scale_to_0th):
    if zero is None:
        zero = np.zeros(arr.shape[0])
    if scale is None:
        scale = np.ones(arr.shape[0])
    if weight is None:
        weight = np.ones(arr.shape[0])

    calc_z, zeros, fun_z = _set_calc_zsw(arr, zero, zero_kw)
    calc_s, scales, fun_s = _set_calc_zsw(arr, scale, scale_kw)
    calc_w, weights, fun_w = _set_calc_zsw(arr, weight)

    if calc_z:
        for i in range(arr.shape[0]):
            zeros.append(fun_z(arr[i, ]))
        zeros = np.array(zeros)

    if calc_s:
        if fun_s == fun_z:
            scales = zeros.copy()
        else:
            for i in range(arr.shape[0]):
                scales.append(fun_s(arr[i, ]))
            scales = np.array(scales)

    if calc_w:  # TODO: Needs update to match IRAF's...
        if fun_w == fun_s:
            weights = scales.copy()
        elif fun_w == fun_z:
            weights = zeros.copy()
        else:
            for i in range(arr.shape[0]):
                weight.append(fun_w(arr[i, ]))
            weights = np.array(weights)

    if zero_to_0th:
        zeros -= zeros[0]

    if scale_to_0th:
        scales /= scales[0]  # So that normalize 1.000 for the 0-th image.

    return zeros, scales, weights


def _setup_reject(arr, mask, nkeep, maxrej, cenfunc):
    ''' Does the common default setting for all rejection algorithms.
    '''
    _arr = arr.copy()
    # NOTE: mask_orig is propagation of input mask && nonfinite(arr)
    _arr[_set_mask(_arr, mask)] = np.nan
    mask_orig = ~np.isfinite(_arr)
    numnan = np.sum(mask_orig, axis=0)
    nkeep, maxrej = _set_keeprej(_arr, nkeep, maxrej, axis=0)
    cenfunc = _set_cenfunc(cenfunc, nameonly=False, nan=True)

    # unsigned 8/16-bit (up to 255/65535) will be enough for num_iter
    # and num_rej, respectively.
    nit = np.ones(_arr.shape[1:], dtype=np.uint8)
    ncombine = _arr.shape[0]
    nrej_old = np.sum(mask_orig, axis=0, dtype=np.uint16)
    n_old = ncombine - nrej_old  # num of remaining pixels

    # Initiate
    low = bn.nanmin(_arr, axis=0)
    upp = bn.nanmax(_arr, axis=0)
    low_new = low.copy()
    upp_new = upp.copy()

    no_nkeep = nkeep is None
    no_maxrej = maxrej is None
    if no_nkeep and no_maxrej:
        mask_nkeep = np.zeros(_arr.shape[1:], dtype=bool)
        mask_maxrej = np.zeros(_arr.shape[1:], dtype=bool)
        mask_pix = np.zeros(_arr.shape[1:], dtype=bool)
    elif not no_nkeep and no_maxrej:
        mask_nkeep = ((ncombine - numnan) < nkeep)
        mask_maxrej = np.zeros(_arr.shape[1:], dtype=bool)
        mask_pix = mask_nkeep.copy()
    elif not no_maxrej and no_nkeep:
        mask_nkeep = np.zeros(_arr.shape[1:], dtype=bool)
        mask_maxrej = (nrej_old > maxrej)
        mask_pix = mask_maxrej.copy()
    else:
        mask_nkeep = ((ncombine - numnan) < nkeep)
        mask_maxrej = (nrej_old > maxrej)
        mask_pix = mask_nkeep | mask_maxrej

    return (_arr,
            [mask_orig, mask_nkeep, mask_maxrej, mask_pix],
            [nkeep, maxrej],
            cenfunc,
            [nit, ncombine, n_old],
            [low, upp, low_new, upp_new]
            )


def write2fits(data, header, output, return_hdu=False, **kwargs):
    hdu = fits.PrimaryHDU(data=data, header=header)
    try:
        hdu.writeto(output, **kwargs)
    except fits.VerifyError:
        print("Try using output_verify='fix' to avoid this error.")
    if return_hdu:
        return hdu


def update_hdr(header, ncombine, imcmb_key, imcmb_val,
               offset_mode=None, offsets=None):
    """ **Inplace** update of the given header
    """
    header["NCOMBINE"] = (ncombine, "Number of combined images")
    if imcmb_key != '':
        header["IMCMBKEY"] = (
            imcmb_key,
            f"Key used in IMCMBiii ('$I': filepath)"
        )
        for i in range(min(999, len(imcmb_val))):
            header[f"IMCMB{i+1:03d}"] = imcmb_val[i]

    if offset_mode is not None:
        header['OFFSTMOD'] = (
            offset_mode,
            "Offset method used for combine."
        )
        for i in range(min(999, len(imcmb_val))):
            header[f"OFFST{i:03d}"] = str(offsets[i, ].tolist())

    # Add "IRAF-TLM" like header key for continuity with IRAF.
    header.set("FITS-TLM",
               value=Time(Time.now(), precision=0).isot,
               comment="UT of last modification of this FITS file",
               after=f"NAXIS{header['NAXIS']}")


def _calculate_step_sizes(x_size, y_size, num_chunks):
    """ Calculate the strides in x and y.
    Notes
    -----
    Calculate the strides in x and y to achieve at least the
    ``num_chunks`` pieces.

    Direct copy from ccdproc:
    https://github.com/astropy/ccdproc/blob/b9ec64dfb59aac1d9ca500ad172c4eb31ec305f8/ccdproc/combiner.py#L500
    """
    # First we try to split only along fast x axis
    xstep = max(1, int(x_size / num_chunks))

    # More chunks are needed only if xstep gives us fewer chunks than
    # requested.
    x_chunks = int(x_size / xstep)

    if x_chunks >= num_chunks:
        ystep = y_size
    else:
        # The x and y loops are nested, so the number of chunks
        # is multiplicative, not additive. Calculate the number
        # of y chunks we need to get at num_chunks.
        y_chunks = int(num_chunks / x_chunks) + 1
        ystep = max(1, int(y_size / y_chunks))

    return xstep, ystep


def _get_combine_shape(sizes, offsets):
    ''' Organize offsets and get shape of combined image.
    '''
    _offsets = np.array(offsets)
    _offsets = np.max(_offsets, axis=0) - offsets  # so that offsets > 0.
    _offsets = np.around(_offsets).astype(int)
    sh_comb = (np.max(np.array(sizes) + _offsets, axis=0)).astype(int)
    return _offsets, sh_comb


def _set_int_dtype(ncombine):
    if ncombine < 255:
        int_dtype = np.uint8
    elif ncombine > 65535:
        int_dtype = np.uint32
    else:
        int_dtype = np.uint16
    return int_dtype


def _set_mask(arr, mask):
    if mask is None:
        mask = np.zeros_like(arr, dtype=bool)
    else:
        mask = np.array(mask, dtype=bool)

    if arr.shape != mask.shape:
        raise ValueError(
            "The input array and mask must have the identical shape. "
            + f"Now arr.shape = {arr.shape} and mask.shape = {mask.shape}."
        )
    return mask


def _set_sigma(sigma, sigma_lower=None, sigma_upper=None):
    sigma = np.atleast_1d(sigma)
    if sigma.shape[0] == 1:
        if sigma_lower is None:
            sigma_lower = float(sigma)
        if sigma_upper is None:
            sigma_upper = float(sigma)
    elif sigma.shape[0] == 2:
        if sigma_lower is None:
            sigma_lower = float(sigma[0])
        if sigma_upper is None:
            sigma_upper = float(sigma[1])
    else:
        raise ValueError("sigma must have shape[0] of 1 or 2.")
    return sigma_lower, sigma_upper


def _set_keeprej(arr, nkeep, maxrej, axis):
    if nkeep < 1:
        nkeep = int(np.around(nkeep*arr.shape[axis]))
    if maxrej is None:
        maxrej = int(arr.shape[axis])
    elif maxrej < 1:
        maxrej = int(np.around(maxrej*arr.shape[axis]))
    return nkeep, maxrej


def _set_minmax(arr, n_minmax, axis):
    n_minmax = np.atleast_1d(n_minmax)
    if n_minmax.shape[0] == 1:
        q_low = float(n_minmax[0])
        q_upp = float(n_minmax[0])
    elif n_minmax.shape[0] == 2:
        q_low = float(n_minmax[0])
        q_upp = float(n_minmax[1])
    else:
        raise ValueError("n_minmax must have shape[0] of 1 or 2.")

    nimages = arr.shape[0]

    if q_low >= 1:  # if given as number of pixels
        q_low = q_low/nimages
    # else: already given as fraction

    if q_upp >= 1:  # if given as number of pixels
        q_upp = q_upp/nimages
    # else: already given as fraction

    if q_low + q_upp > 1:
        raise ValueError("min/max rejection more than images!")

    return q_low, q_upp


def _set_thresh_mask(arr, mask, thresholds, update_mask=True):
    if (thresholds[0] != -np.inf) and (thresholds[1] != np.inf):
        mask_thresh = (arr < thresholds[0]) | (arr > thresholds[1])
        if update_mask:
            mask |= mask_thresh
    elif (thresholds[0] == -np.inf):
        mask_thresh = (arr > thresholds[1])
        if update_mask:
            mask |= mask_thresh
    elif (thresholds[1] == np.inf):
        mask_thresh = (arr < thresholds[0])
        if update_mask:
            mask |= mask_thresh
    else :
        mask_thresh = np.zeros(arr.shape).astype(bool)
        # no need to update _mask

    return mask_thresh


def _set_cenfunc(cenfunc, shorten=False, nameonly=True, nan=True):
    if cenfunc in ['med', 'medi', 'median']:
        if nameonly:
            cen = 'med' if shorten else 'median'
        else:
            cen = bn.nanmedian if nan else bn.median
    elif cenfunc in ['avg', 'average', 'mean']:
        if nameonly:
            cen = 'avg' if shorten else 'average'
        else:
            cen = bn.nanmean if nan else np.mean
    elif cenfunc in ['lmed', 'lmd', 'lmedian']:
        if nameonly:
            cen = 'lmd' if shorten else 'lmedian'
        else:
            cen = nanlmedian if nan else lmedian
    else:
        raise ValueError('cenfunc not understood')

    return cen


def _set_combfunc(combfunc, shorten=False, nameonly=True, nan=True):
    ''' Identical to _set_cenfunc, except 'sum' is allowed.
    '''
    if combfunc in ['med', 'medi', 'median']:
        if nameonly:
            comb = 'med' if shorten else 'median'
        else:
            comb = bn.nanmedian if nan else bn.median
    elif combfunc in ['avg', 'average', 'mean']:
        if nameonly:
            comb = 'avg' if shorten else 'average'
        else:
            comb = bn.nanmean if nan else np.mean
    elif combfunc in ['sum']:
        if nameonly:
            comb = 'sum'
        else:
            comb = bn.nansum if nan else np.sum
    elif combfunc in ['lmed', 'lmd', 'lmedian']:
        if nameonly:
            comb = 'lmd' if shorten else 'lmedian'
        else:
            comb = nanlmedian if nan else lmedian
    else:
        raise ValueError('combfunc not understood')

    return comb


def _set_reject_name(reject):
    if reject is None:
        return reject

    rej = reject.lower()
    if rej in ['sig', 'sc', 'sigclip', 'sigma', 'sigma clip', 'sigmaclip']:
        rej = 'sigclip'
    elif rej in ['mm', 'minmax']:
        rej = 'minmax'
    elif rej in ['ccd', 'ccdclip', 'ccdc']:
        rej = 'ccdclip'
    elif rej in ['pclip', 'pc', 'percentile']:
        rej = 'pclip'
    return rej


# TODO: add sigma-clipped mean, med, std as scale, zero, or weight.
def _set_calc_zsw(arr, scale_zero_weight, zsw_kw={}):
    if isinstance(scale_zero_weight, str):
        zswstr = scale_zero_weight.lower()
        calc_zsw = True
        zsw = []
        if zswstr in ['avg', 'average', 'mean']:
            calcfun = bn.nanmean
        elif zswstr in ['med', 'medi', 'median']:
            calcfun = bn.nanmedian
        elif zswstr in ['avg_sc', 'average_sc', 'mean_sc']:
            _ = zsw_kw.pop('axis', None)  # if exist ``axis``, remove it.

            def calcfun(x):
                return sigma_clipped_stats(x, **zsw_kw)[0]
        elif zswstr in ['med_sc', 'medi_sc', 'median_sc']:
            _ = zsw_kw.pop('axis', None)  # if exist ``axis``, remove it.

            def calcfun(x):
                return sigma_clipped_stats(x, **zsw_kw)[1]
    else:
        zsw = np.array(scale_zero_weight)
        if zsw.size == arr.shape[0]:
            zsw = zsw.flatten()
        else:
            raise ValueError(
                "If scale/zero/weight are array-like, they must be of size "
                + "identical to arr.shape[0] (number of images to combine)."
            )
        calc_zsw = False
        zsw = np.array(scale_zero_weight)
        calcfun = None
    return calc_zsw, zsw, calcfun


def slice_from_string(string, fits_convention=False):
    """ Convert a string to a tuple of slices.
    Direct copy from `ccdproc`_

    .. _ccdproc: https://github.com/astropy/ccdproc/blob/b9ec64dfb59aac1d9ca500ad172c4eb31ec305f8/ccdproc/utils/slices.py#L10

    Parameters
    ----------
    string : str
        A string that can be converted to a slice.
    fits_convention : bool, optional
        If True, assume the input string follows the FITS convention for
        indexing: the indexing is one-based (not zero-based) and the first
        axis is that which changes most rapidly as the index increases.

    Returns
    -------
    slice_tuple : tuple of slice objects
        A tuple able to be used to index a numpy.array

    Notes
    -----
    The ``string`` argument can be anything that would work as a valid way to
    slice an array in Numpy. It must be enclosed in matching brackets; all
    spaces are stripped from the string before processing.

    Examples
    --------
    >>> import numpy as np
    >>> arr1d = np.arange(5)
    >>> a_slice = slice_from_string('[2:5]')
    >>> arr1d[a_slice]
    array([2, 3, 4])
    >>> a_slice = slice_from_string('[ : : -2] ')
    >>> arr1d[a_slice]
    array([4, 2, 0])
    >>> arr2d = np.array([arr1d, arr1d + 5, arr1d + 10])
    >>> arr2d
    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14]])
    >>> a_slice = slice_from_string('[1:-1, 0:4:2]')
    >>> arr2d[a_slice]
    array([[5, 7]])
    >>> a_slice = slice_from_string('[0:2,0:3]')
    >>> arr2d[a_slice]
    array([[0, 1, 2],
           [5, 6, 7]])
    """
    no_space = string.replace(' ', '')

    if not no_space:
        return ()

    if not (no_space.startswith('[') and no_space.endswith(']')):
        raise ValueError('Slice string must be enclosed in square brackets.')

    no_space = no_space.strip('[]')
    if fits_convention:
        # Special cases first
        # Flip dimension, with step
        no_space = no_space.replace('-*:', '::-')
        # Flip dimension
        no_space = no_space.replace('-*', '::-1')
        # Normal wildcard
        no_space = no_space.replace('*', ':')
    string_slices = no_space.split(',')
    slices = []
    for string_slice in string_slices:
        slice_args = [int(arg) if arg else None
                      for arg in string_slice.split(':')]
        a_slice = slice(*slice_args)
        slices.append(a_slice)

    if fits_convention:
        slices = _defitsify_slice(slices)

    return tuple(slices)


def _defitsify_slice(slices):
    """ Convert a FITS-style slice specification into a python slice.
    Direct copy from https://github.com/astropy/ccdproc/blob/b9ec64dfb59aac1d9ca500ad172c4eb31ec305f8/ccdproc/utils/slices.py#L1
    This means two things:
    + Subtract 1 from starting index because in the FITS
      specification arrays are one-based.
    + Do **not** subtract 1 from the ending index because the python
      convention for a slice is for the last value to be one less than the
      stop value. In other words, this subtraction is already built into
      python.
    + Reverse the order of the slices, because the FITS specification dictates
      that the first axis is the one along which the index varies most rapidly
      (aka FORTRAN order).
    """

    python_slice = []
    for a_slice in slices[::-1]:
        new_start = a_slice.start - 1 if a_slice.start is not None else None
        if new_start is not None and new_start < 0:
            raise ValueError("Smallest permissible FITS index is 1")
        if a_slice.stop is not None and a_slice.stop < 0:
            raise ValueError("Negative final index not allowed for FITS slice")
        new_slice = slice(new_start, a_slice.stop, a_slice.step)
        if (a_slice.start is not None and a_slice.stop is not None and
                a_slice.start > a_slice.stop):
            # FITS use a positive step index when dimension are inverted
            new_step = -1 if a_slice.step is None else -a_slice.step
            # Special case to prevent -1 as slice stop value
            new_stop = None if a_slice.stop == 1 else a_slice.stop-2
            new_slice = slice(new_start, new_stop, new_step)
        python_slice.append(new_slice)

    return python_slice


# NOTE: I dunno the reason but this is ~ 3 times slower than
# astropy.stats.sigma_clipped_stats calcualtion.
def sigmaclip_inf(a, sigma_lower=3., sigma_upper=3.,
                  cenfunc='median', ddof=1, outfunc='median'):
    """ Infinite loop sigma-clip.

    copy from `scipy`_.

    .. _scipy: https://github.com/scipy/scipy/blob/4c0fd79391e3b2ec2738bf85bb5dab366dcd12e4/scipy/stats/stats.py#L3159-L3225
    """
    # bn.median is ~ 2x fater than np.median
    cenfunc = _set_cenfunc(cenfunc, nameonly=False, nan=False)
    outfunc = _set_cenfunc(outfunc, nameonly=False, nan=False)

    c = np.asarray(a).ravel()
    c = c[~np.isnan(c)]
    delta = 1
    while delta:
        std = np.std(c, ddof=ddof)
        cen = cenfunc(c)
        size = c.size
        critlower = cen - std * sigma_lower
        critupper = cen + std * sigma_upper
        c = c[(c >= critlower) & (c <= critupper)]
        delta = size - c.size

    return outfunc(c)
