import warnings
import glob

import bottleneck as bn
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.time import Time
from astropy.nddata import CCDData, StdDevUncertainty, VarianceUncertainty
from astropy.wcs import WCS

from .numpy_util import lmedian, nanlmedian

__all__ = [
    'load_fits', 'load_ccd', 'filelist', 'write2fits', 'update_hdr',
    'add_to_header', 'do_zs', 'get_zsw',
    '_get_combine_shape', '_set_int_dtype', '_get_dtype_limits',
    '_setup_reject',
    '_set_mask', '_set_sigma', '_set_keeprej', '_set_minmax',
    '_set_thresh_mask', '_set_gain_rdns',
    '_set_cenfunc', '_set_combfunc',
    '_set_reject_name', "slice_from_string"
]


def load_fits(_fpath, ext=0, ext_uncertainty=None, ext_mask=None,
              uncertainty_type='stddev', use_cfitsio=True):
    ''' Load FITS file using either fitsio or load_ccd.

    Returns
    -------
    _data : ndarray
        The data (from ``ext``).

    _var : ndarray or `None`
        The variance array (from ``ext_uncertainty``). It is **NOT**
        astropy's NDData format.

    _mask : ndarray
        The mask array (from ``ext_mask``). If no mask extension is
        found, an ndarray with identical shape to ``_data`` filled with
        `False` will be returned, **NOT** `None`.
    '''
    if use_cfitsio:
        import fitsio

        hdul = fitsio.FITS(_fpath)
        if str(ext).isnumeric():
            _data = hdul[ext].read()
        else:
            raise ValueError(
                "Currently ext in str is not supported "
                + "with fitsio. Try integer or use_cfitsio=False.")

        if ext_uncertainty is not None:
            if str(ext_uncertainty).isnumeric():
                _var = hdul[ext_uncertainty].read()
                if uncertainty_type in ['stddev', 'stdev', 'std']:
                    _var = _var**2
                elif uncertainty_type not in ['var', 'vari', 'variance']:
                    raise ValueError("uncertainty_type not understood.")
            else:
                raise ValueError(
                    "Currently ext_uncertainty in str is not supported "
                    + "wwith fitsio. Try integer or use_cfitsio=False.")
        else:
            _var = None

        if ext_mask is not None:
            if str(ext_mask).isnumeric():
                _mask = hdul[ext_mask].read()
            else:
                raise ValueError(
                    "Currently ext_mask in str is not supported "
                    + "with fitsio. Try integer or use_cfitsio=False.")
        else:
            _mask = None

    else:
        ccd = load_ccd(_fpath, extension=ext, memmap=False)
        _data = ccd.data
        _mask = ccd.mask
        if ccd.uncertainty is None:
            _var = None
        elif isinstance(ccd.uncertainty, StdDevUncertainty):
            _var = ccd.uncertainty.array**2
        elif isinstance(ccd.uncertainty, VarianceUncertainty):
            _var = ccd.uncertainty.array
        else:
            raise ValueError(
                f"CCDData.read({_fpath}) gave a strange uncertainty.."
            )

    if _mask is None:
        _mask = np.zeros(_data.shape, dtype=bool)

    if use_cfitsio:
        hdul.close()
    else:
        del ccd, _data, _mask

    return _data, _var, _mask


# FIXME: Remove it in the future.
def load_ccd(path, extension=0, unit=None, hdu_uncertainty="UNCERT",
             use_wcs=True, hdu_mask='MASK', hdu_flags=None,
             key_uncertainty_type='UTYPE', prefer_bunit=True, memmap=False,
             **kwd):
    '''Copy from ysfitsutilpy
    https://github.com/ysBach/ysfitsutilpy/blob/50437167e1ad8b62c40dc7436c836254fb8dba37/ysfitsutilpy/misc.py#L100
    remove it when astropy updated:
    Note
    ----
    CCDData.read cannot read TPV WCS:
    https://github.com/astropy/astropy/issues/7650
    Also memory map must be set False to avoid memory problem
    https://github.com/astropy/astropy/issues/9096
    Plus, WCS info from astrometry.net solve-field sometimes not
    understood by CCDData.read....
    2020-05-31 16:39:51 (KST: GMT+09:00) ysBach
    '''
    reader_kw = dict(hdu=extension, hdu_uncertainty=hdu_uncertainty,
                     hdu_mask=hdu_mask, hdu_flags=hdu_flags,
                     key_uncertainty_type=key_uncertainty_type,
                     memmap=memmap, **kwd)
    if use_wcs:
        hdr = fits.getheader(path)
        reader_kw["wcs"] = WCS(hdr)
        del hdr

    if not prefer_bunit:  # prefer user's input
        ccd = CCDData.read(path, unit=unit, **reader_kw)
    else:
        ccd = CCDData.read(path, unit=None, **reader_kw)

    return ccd


def filelist(fpattern, fpaths=None):
    if fpaths is None:
        fpaths = glob.glob(fpattern)
    fpaths = list(fpaths)
    fpaths.sort()
    return fpaths


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
    # Also note that both of the below show nearly identical timing
    # (https://stackoverflow.com/a/45895371/7199629)
    #   (data3d_nan.T / scale).T
    #   np.array([data3d_nan[i, ] / _s for i, _s in enumerate(scale)])
    # 7.58 +/- 0.3 ms, 8.46 +/- 1.88 ms, respectively. Though the former
    # is concise, latter is more explicit, so I used the latter.
    return arr


def get_zsw(arr, zero, scale, weight, zero_kw, scale_kw,
            zero_to_0th, scale_to_0th, zero_section, scale_section):
    # TODO: add sigma-clipped mean, med, std as scale, zero, or weight.
    def _nanfun2nonnan(fun):
        '''
        Returns
        -------
        nonnan_function: The function without NaN policy
        check_isfinite: Whether the ``isfinite`` must be tested.
        '''
        if fun == bn.nanmean:
            return np.mean
        elif fun == bn.nanmedian:
            # There IS bn.median, but that doesn't accept tuple axis...
            return np.median
        elif fun == bn.nansum:
            return np.sum
        else:
            return fun

    def _set_calc_zsw(arr, zero_scale_weight, zsw_kw={}):
        if isinstance(zero_scale_weight, str):
            zswstr = zero_scale_weight.lower()
            calc_zsw = True
            zsw = []
            if zswstr in ['avg', 'average', 'mean']:
                calcfun = bn.nanmean
            elif zswstr in ['med', 'medi', 'median']:
                calcfun = bn.nanmedian
            elif zswstr in ['avg_sc', 'average_sc', 'mean_sc']:
                _ = zsw_kw.pop('axis', None)  # if exist ``axis``, remove it.
                calcfun = lambda x, zsw_kw: sigma_clipped_stats(x, **zsw_kw)[0]
            elif zswstr in ['med_sc', 'medi_sc', 'median_sc']:
                _ = zsw_kw.pop('axis', None)  # if exist ``axis``, remove it.
                calcfun = lambda x, zsw_kw: sigma_clipped_stats(x, **zsw_kw)[1]
            else:
                raise ValueError(
                    f"zero/scale/weight ({zero_scale_weight}) not understood")

        else:
            zsw = np.array(zero_scale_weight)
            if zsw.size == arr.shape[0]:
                zsw = zsw.flatten()
            else:
                raise ValueError(
                    "If scale/zero/weight are array-like, they must be of "
                    + "size = arr.shape[0] (number of images to combine)."
                )
            calc_zsw = False
            zsw = np.array(zero_scale_weight)
            calcfun = None
        return calc_zsw, zsw, calcfun

    def _calc_zsw(fun, arr, axis, section=None):
        fun_simple = _nanfun2nonnan(fun)
        if section is not None:
            sl = slice_from_string(section, fits_convention=True)
        else:
            sl = [slice(None)]*(arr.ndim - 1)

        try:
            # If converted to numpy, this must work without error:
            zsw = fun_simple(arr[(slice(None), *sl)], axis=simple_axis)
            redo = ~np.all(np.isfinite(zsw))
        except TypeError:  # does not accept tuple axis
            redo = True

        if redo:
            zsw = []
            for i in range(arr.shape[0]):
                zsw.append(fun(arr[(i, *sl)]))

        return np.array(zsw)

    if zero is None:
        zeros = np.zeros(arr.shape[0])
        calc_z = False
        fun_z = None
    else:
        calc_z, zeros, fun_z = _set_calc_zsw(arr, zero, zero_kw)

    if scale is None:
        scales = np.ones(arr.shape[0])
        calc_s = False
        fun_s = None
    else:
        calc_s, scales, fun_s = _set_calc_zsw(arr, scale, scale_kw)

    if weight is None:
        weights = np.ones(arr.shape[0])
        calc_w = False
        fun_w = None
    else:
        calc_w, weights, fun_w = _set_calc_zsw(arr, weight)

    simple_axis = tuple(np.arange(arr.ndim)[1:])

    if calc_z:
        zeros = _calc_zsw(fun_z, arr, simple_axis, section=zero_section)

    if calc_s:
        if fun_s is not None and fun_z is not None and fun_s == fun_z:
            scales = zeros.copy()
        else:
            scales = _calc_zsw(fun_s, arr, simple_axis, section=scale_section)

    if calc_w:  # TODO: Needs update to match IRAF's...
        if fun_w is not None and fun_s is not None and fun_w == fun_s:
            weights = scales.copy()
        elif fun_w is not None and fun_z is not None and fun_w == fun_z:
            weights = zeros.copy()
        else:
            weights = _calc_zsw(fun_w, arr, simple_axis, section=scale_section)

    if zero_to_0th:
        zeros -= zeros[0]

    if scale_to_0th:
        scales /= scales[0]  # So that normalize 1.000 for the 0-th image.

    return zeros, scales, weights


def _setup_reject(arr, mask, nkeep, maxrej, cenfunc):
    ''' Does the common default setting for all rejection algorithms.
    '''
    _arr = arr.copy()
    # NOTE: mask_nan is propagation of input mask && nonfinite(arr)
    _arr[_set_mask(_arr, mask)] = np.nan
    mask_nan = ~np.isfinite(_arr)

    numnan = np.count_nonzero(mask_nan, axis=0)
    nkeep, maxrej = _set_keeprej(_arr, nkeep, maxrej, axis=0)
    cenfunc = _set_cenfunc(cenfunc, nameonly=False, nan=True)

    ncombine = _arr.shape[0]
    int_dtype = _set_int_dtype(ncombine)
    # n_iteration : in uint8
    nit = np.ones(_arr.shape[1:], dtype=np.uint8)
    # n_rejected/n_finite_old : all in int_dtype determined by _set_int_dtype
    n_nan = np.count_nonzero(mask_nan, axis=0).astype(int_dtype)
    n_finite_old = ncombine - n_nan  # num of remaining pixels

    # Initiate with min/max, not something like std.
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
        mask_maxrej = (n_nan > maxrej)
        mask_pix = mask_maxrej.copy()
    else:
        mask_nkeep = ((ncombine - numnan) < nkeep)
        mask_maxrej = (n_nan > maxrej)
        mask_pix = mask_nkeep | mask_maxrej

    return (_arr,
            [mask_nan, mask_nkeep, mask_maxrej, mask_pix],
            [nkeep, maxrej],
            cenfunc,
            [nit, ncombine, n_finite_old],
            [low, upp, low_new, upp_new]
            )


def write2fits(data, header, output, return_ccd=False, **kwargs):
    try:
        unit = header['BUNIT']
    except (KeyError, IndexError):
        unit = 'adu'
    ccd = CCDData(data=data, header=header, unit=unit)

    try:
        ccd.write(output, **kwargs)
    except fits.VerifyError:
        print("Try using output_verify='fix' to avoid this error.")
    if return_ccd:
        return ccd


def update_hdr(header, ncombine, imcmb_key, imcmb_val,
               offset_mode=None, offsets=None):
    """ **Inplace** update of the given header
    """
    header["NCOMBINE"] = (ncombine, "Number of combined images")
    if imcmb_key != '':
        header["IMCMBKEY"] = (
            imcmb_key,
            "Key used in IMCMBiii ('$I': filepath)"
        )
        for i in range(min(999, len(imcmb_val))):
            header[f"IMCMB{i+1:03d}"] = imcmb_val[i]

    if offset_mode is not None:
        header['OFFSTMOD'] = (
            offset_mode,
            "Offset method used for combine."
        )
        for i in range(min(999, len(imcmb_val))):
            header[f"OFFST{i:03d}"] = str(offsets[i, ][::-1].tolist())

    # Add "IRAF-TLM" like header key for continuity with IRAF.
    header.set("FITS-TLM",
               value=Time(Time.now(), precision=0).isot,
               comment="UT of last modification of this FITS file",
               after=f"NAXIS{header['NAXIS']}")


def str_now(precision=3, fmt="{:.>72s}", t_ref=None,
            dt_fmt="(dt = {:.3f} s)", return_time=False):
    ''' Get stringfied time now in UT ISOT format.
    Parameters
    ----------
    precision : int, optional.
        The precision of the isot format time.
    fmt : str, optional.
        The Python 3 format string to format the time.
        Examples:
          * ``"{:s}"``: plain time ``2020-01-01T01:01:01.23``
          * ``"({:s})"``: plain time in parentheses
            ``(2020-01-01T01:01:01.23)``
          * ``"{:_^72s}"``: center align, filling with ``_``.
    t_ref : Time, optional.
        The reference time. If not ``None``, delta time is calculated.
    dt_fmt : str, optional.
        The Python 3 format string to format the delta time.
    return_time : bool, optional.
        Whether to return the time at the start of this function and the
        delta time (``dt``), as well as the time information string. If
        ``t_ref`` is ``None``, ``dt`` is automatically set to ``None``.
    '''
    now = Time(Time.now(), precision=precision)
    timestr = now.isot
    if t_ref is not None:
        dt = (now - Time(t_ref)).sec  # float in seconds unit
        timestr = dt_fmt.format(dt) + " " + timestr
    else:
        dt = None

    if return_time:
        return fmt.format(timestr), now, dt
    else:
        return fmt.format(timestr)


def add_to_header(header, histcomm, s, precision=3,
                  fmt="{:.>72s}", t_ref=None, dt_fmt="(dt = {:.3f} s)",
                  verbose=False):
    ''' Automatically add timestamp as well as history string
    Parameters
    ----------
    header : Header
        The header.
    histcomm : str in ['h', 'hist', 'history', 'c', 'comm', 'comment']
        Whether to add history or comment.
    s : str or list of str
        The string to add as history or comment.
    precision : int, optional.
        The precision of the isot format time.
    fmt : str, None, optional.
        The Python 3 format string to format the time in the header.
        If ``None``, the timestamp string will not be added.
        Examples:
          * ``"{:s}"``: plain time ``2020-01-01T01:01:01.23``
          * ``"({:s})"``: plain time in parentheses
            ``(2020-01-01T01:01:01.23)``
          * ``"{:_^72s}"``: center align, filling with ``_``.
    t_ref : Time
        The reference time. If not ``None``, delta time is calculated.
    dt_fmt : str, optional.
        The Python 3 format string to format the delta time in the
        header.
    verbose : bool, optional.
        Whether to print the same information on the output terminal.
    verbose_fmt : str, optional.
        The Python 3 format string to format the time in the terminal.
    '''
    if isinstance(s, str):
        s = [s]

    if histcomm.lower() in ['h', 'hist', 'history']:
        for _s in s:
            header.add_history(_s)
            if verbose:
                print(f"HISTORY {_s}")
        if fmt is not None:
            timestr = str_now(precision=precision, fmt=fmt,
                              t_ref=t_ref, dt_fmt=dt_fmt)
            header.add_history(timestr)
            if verbose:
                print(f"HISTORY {timestr}")

    elif histcomm.lower() in ['c', 'comm', 'comment']:
        for _s in s:
            header.add_comment(s)
            if verbose:
                print(f"COMMENT {_s}")
        if fmt is not None:
            timestr = str_now(precision=precision, fmt=fmt,
                              t_ref=t_ref, dt_fmt=dt_fmt)
            header.add_comment(timestr)
            if verbose:
                print(f"COMMENT {timestr}")


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


def _get_dtype_limits(dtype):
    try:
        info = np.iinfo(dtype)
    except ValueError:
        # I don't use np.inf, because of a fear that some functions,
        # e.g., bn.partition mayy not work for these, as well as np.nan.
        info = np.finfo(dtype)
    return (info.min, info.max)


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
    if nkeep is None:
        nkeep = 0
    elif nkeep < 1:
        nkeep = np.around(nkeep*arr.shape[axis])

    if maxrej is None:
        maxrej = arr.shape[axis]
    elif maxrej < 1:
        maxrej = np.around(maxrej*arr.shape[axis])

    return int(nkeep), int(maxrej)


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


def _set_gain_rdns(gain_or_rdnoise, ncombine, dtype='float32'):
    extract = False
    if isinstance(gain_or_rdnoise, str):
        extract = True
        arr = np.ones(ncombine, dtype=dtype)
    else:
        arr = np.array(gain_or_rdnoise).astype(dtype)
        if arr.size == ncombine:
            arr = arr.ravel()
            if not np.all(np.isfinite(arr)):
                raise ValueError("gain or rdnoise contains NaN")
        elif arr.size == 1:
            if not np.isfinite(arr):
                raise ValueError("gain or rdnoise contains NaN")
            arr = arr*np.ones(ncombine, dtype=dtype)
        else:
            raise ValueError("gain or rdnoise size not equal to ncombine.")
    return extract, arr


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
