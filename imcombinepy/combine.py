from pathlib import Path

import bottleneck as bn
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import CCDData

from .reject import sigclip_mask, ccdclip_mask
from .util import (_get_combine_shape, _set_cenfunc, _set_combfunc,
                   _set_gain_rdns, _set_int_dtype, _set_keeprej, _set_mask,
                   _set_reject_name, _set_sigma, _set_thresh_mask, do_zs,
                   filelist, get_zsw, update_hdr, write2fits, load_ccd)
from . import docstrings

__all__ = ["fitscombine", "ndcombine"]

'''
removed : headers, project, masktype, maskvalue, sigscale, grow
partial removal:
    * combine in ["quadrature", "nmodel"]
replaced
    * reject in ["crreject", "avsigclip"] --> ccdclip with certain params
    * offsets in ["grid", <filename>]  --> offsets in ndarray

bpmasks                : ?
rejmask                : output_mask
nrejmasks              : output_nrej
expmasks               : Should I implement???
sigma                  : output_sigma
outtype                : dtype
outlimits              : fits_section
expname                : exposure_key

# ALGORITHM PARAMETERS ====================================================== #
lthreshold, hthreshold : thresholds (tuple)
nlow      , nhigh      : n_minmax (tuple)
nkeep                  : nkeep & maxrej
                        (IRAF nkeep > 0 && < 0 case, resp.)
mclip                  : cenfunc
lsigma    , hsigma     : sigma uple
'''


def fitscombine(
        fpaths=None, fpattern=None, mask=None, ext=0,
        fits_section=None,
        blank=np.nan,
        offsets=None,
        thresholds=[-np.inf, np.inf],
        zero=None, scale=None, weight=None, statsec=None,
        zero_kw={'cenfunc': 'median', 'stdfunc': 'std', 'std_ddof': 1},
        scale_kw={'cenfunc': 'median', 'stdfunc': 'std', 'std_ddof': 1},
        zero_to_0th=True, scale_to_0th=True,
        zero_section=None, scale_section=None,
        reject=None,
        cenfunc='median',
        sigma=[3., 3.], maxiters=50, ddof=1, nkeep=1, maxrej=None,
        n_minmax=[1, 1],
        rdnoise=0., gain=1., snoise=0.,
        pclip=-0.5,
        logfile=None,
        combine='average',
        dtype='float32',
        irafmode=True,
        memlimit=2.5e+9,
        verbose=False,
        full=False,
        imcmb_key='$I',
        exposure_key="EXPTIME",
        output=None, output_mask=None, output_nrej=None,
        output_std=None, output_low=None, output_upp=None,
        output_rejcode=None, return_dict=False,
        **kwargs
):
    if verbose:
        print("Organizing", end='... ')

    if (fpaths is not None) + (fpattern is not None) != 1:
        raise ValueError("Give one and only one of fpaths/fpattern.")

    fpaths = filelist(fpattern, fpaths=fpaths)
    ncombine = len(fpaths)
    reject = _set_reject_name(reject)
    int_dtype = _set_int_dtype(ncombine)

    # == check if we should care about memory =============================== #
    # It usually takes < 1ms for hundreds of files
    # What we get here is the lower bound of the total memory used.
    # Even if chop_load is False, we later may have to use chopping when
    # combine. See below.
    # chop_load = False
    fsize_tot = 0
    for fpath in fpaths:
        fsize_tot += Path(fpath).stat().st_size
    # if fsize_tot > memlimit:
    #     chop_load = True
    # ----------------------------------------------------------------------- #

    hdr0 = fits.getheader(Path(fpaths[0]), ext=ext)
    ndim = hdr0['NAXIS']
    # N x ndim. sizes[i, :] = images[i].shape
    sizes = np.ones((ncombine, ndim), dtype=int)

    imcmb_val = []
    extract_exptime = False
    if isinstance(scale, str):
        if scale.lower() in ["exp", "expos", "exposure", "exptime"]:
            extract_exptime = True
            scale = np.ones(ncombine, dtype=dtype)

    if reject == 'ccdclip':
        extract_gain, gns = _set_gain_rdns(gain, ncombine, dtype=dtype)
        extract_rdnoise, rds = _set_gain_rdns(rdnoise, ncombine, dtype=dtype)
        extract_snoise, sns = _set_gain_rdns(snoise, ncombine, dtype=dtype)
    else:
        extract_gain, gns = False, 1
        extract_rdnoise, rds = False, 0
        extract_snoise, sns = False, 0

    # == organize offsets =================================================== #
    # TODO: if offsets is None and ``fsize_tot`` << memlimit, why not
    # just load all data here?
    # initialize
    use_wcs, use_phy = False, False
    if isinstance(offsets, str):
        if offsets.lower() in ['world', 'wcs']:
            w_ref = WCS(hdr0)
            cen_ref = np.array([hdr0[f'NAXIS{i+1}']/2 for i in range(ndim)])
            use_wcs = True
            offset_mode = "WCS"
            offsets = np.zeros((ncombine, ndim))
        elif offsets.lower() in ['physical', 'phys', 'phy']:
            use_phy = True
            offset_mode = "Physical"
            offsets = np.zeros((ncombine, ndim))
        else:
            raise ValueError("offsets not understood.")
    elif offsets is None:
        offset_mode = None
        offsets = np.zeros((ncombine, ndim))
    else:
        if offsets.shape[0] != ncombine:
            raise ValueError("offset.shape[0] must be num(images)")
        offset_mode = "User"
        offsets = np.array(offsets)

    # iterate over files
    for i, fpath in enumerate(fpaths):
        fpath = Path(fpath)
        hdr = fits.getheader(fpath, ext=ext)
        if imcmb_key != '':
            if imcmb_key == "$I":
                imcmb_val.append(fpath.name)
            else:
                try:
                    imcmb_val.append(hdr[imcmb_key])
                except KeyError:
                    imcmb_val.append('')

        if extract_exptime:
            scale[i] = float(hdr[exposure_key])

        if extract_gain:
            gns[i] = float(hdr[gain])  # gain is given as header key

        if extract_rdnoise:
            rds[i] = float(hdr[rdnoise])  # rdnoise is given as header key

        if extract_snoise:
            sns[i] = float(hdr[snoise])  # snoise is given as header key

        if hdr['NAXIS'] != ndim:
            raise ValueError(
                "All FITS files must have the identical dimensionality, "
                + "though they can have different sizes."
            )

        # Update offsets if WCS or Physical should be used
        if use_wcs:
            # Code if using WCS, which may be much slower (but accurate?)
            # Find the center's pixel position in w_ref, in nearest int
            w = WCS(hdr)
            cen = [hdr[f'NAXIS{i+1}']/2 for i in range(ndim)]
            cen_coo = w.all_pix2world(*cen, 0)
            cen = np.around(w_ref.all_world2pix(*cen_coo, 0)).astype(int)
            # NOTE: the indexing in python is [z, y, x] order!!
            offsets[i, ] = (cen_ref - cen)[::-1]
            # For IRAF-like calculation, use
            # offsets[i, ] = [hdr[f'CRPIX{i}'] for i in range(ndim, 0, -1)]
        elif use_phy:
            # NOTE: the indexing in python is [z, y, x] order!!
            offsets[i, ] = [hdr[f'LTV{i}'] for i in range(ndim, 0, -1)]

        # NOTE: the indexing in python is [z, y, x] order!!
        sizes[i, ] = [int(hdr[f'NAXIS{i}']) for i in range(ndim, 0, -1)]

        del hdr
        # ------------------------------------------------------------------- #

    # == Check the size of the temporary array for combination ============== #
    offsets, sh_comb = _get_combine_shape(sizes, offsets)

    # Size of (N+1)-D array before combining along axis=0
    stacksize = np.prod((ncombine, *sh_comb))*(np.dtype(dtype).itemsize)
    # # size estimated by full-stacked array (1st term) plus combined
    # # image (1/ncombine), low and upp bounds (each 1/ncombine), mask
    # # (bool8), niteration (int8), and code(int8).
    # temp_arr_size = stacksize*(1 + 1/ncombine*4)

    # Copied from ccdproc v 2.0.1
    # https://github.com/astropy/ccdproc/blob/b9ec64dfb59aac1d9ca500ad172c4eb31ec305f8/ccdproc/combiner.py#L710
    # Set a memory use factor based on profiling
    combmeth = _set_combfunc(combine)
    memory_factor = 3 if combmeth == "median" else 2
    memory_factor *= 1.5
    mem_req = memory_factor * stacksize
    num_chunk = int(mem_req / memlimit) + 1

    if num_chunk > 1:
        raise ValueError(
            "Currently chunked combine is not supporte T__T. "
            + "Please try increasing memlimit to > {:.1e}, ".format(mem_req)
            + "or reduce number of frames, or use combine='avg' than 'median'."
        )
    if verbose:
        print("Done.")
        if num_chunk > 1:
            print(f"memlimit reached: Split combine by {num_chunk} chunks.")

    # TODO: make chunking

    # == Setup offset-ed array ============================================== #
    # NOTE: Using NaN does not set array with dtype of int... Any solution?
    arr_full = np.nan*np.zeros(shape=(ncombine, *sh_comb), dtype=dtype)
    mask_full = np.zeros(shape=(ncombine, *sh_comb), dtype=bool)
    zeros = np.zeros(shape=ncombine)
    scales = np.zeros(shape=ncombine)
    weights = np.zeros(shape=ncombine)

    for i, (_fpath, _offset, _size) in enumerate(zip(fpaths,
                                                     offsets,
                                                     sizes)):
        # import os
        # import psutil
        # import sys
        # import gc

        # process = psutil.Process(os.getpid())
        # print("0: ", process.memory_info().rss/1.e+9)  # in bytes

        # -- Set slice ------------------------------------------------------ #
        slices = [i]
        # offset & size at each j-th dimension axis
        for offset_j, size_j in zip(_offset, _size):
            slices.append(slice(offset_j, offset_j + size_j, None))

        # -- Set mask ------------------------------------------------------- #
        # process = psutil.Process(os.getpid())
        # print("1: ", process.memory_info().rss/1.e+9)  # in bytes

        ccd = load_ccd(_fpath, extension=ext, memmap=False)
        _data = ccd.data

        # process = psutil.Process(os.getpid())
        # print("1-1: ", process.memory_info().rss/1.e+9)  # in bytes
        _mask = ccd.mask
        # process = psutil.Process(os.getpid())
        # print("1-2: ", process.memory_info().rss/1.e+9)  # in bytes

        if _mask is None:
            _mask = np.zeros(_data.shape, dtype=bool)
        # process = psutil.Process(os.getpid())
        # print("1-3: ", process.memory_info().rss/1.e+9)  # in bytes

        if mask is not None:
            _mask |= mask[i, ]

        # process = psutil.Process(os.getpid())
        # print("2: ", process.memory_info().rss/1.e+9)  # in bytes
        # local_vars = list(locals().items())
        # for var, obj in local_vars:
        #     print(var, sys.getsizeof(obj))

        # -- zero and scale --------------------------------------------- #
        # better to calculate here than from full array, as the
        # latter may contain too many NaNs due to offest shifting.
        # TODO: let get_zsw to get functionals for zsw, so _set_calc_zsw
        # will not be repeateded for every iteration.
        _data_fake = np.array(_data[None, :])  # make a fake (N+1)-D array
        _z, _s, _w = get_zsw(
            arr=_data_fake,
            zero=zero,
            scale=scale,
            weight=weight,
            zero_kw=zero_kw,
            scale_kw=scale_kw,
            zero_to_0th=False,   # to retain original zero
            scale_to_0th=False,  # to retain original scale
            zero_section=zero_section,
            scale_section=scale_section
        )
        zeros[i] = _z[0]
        scales[i] = _s[0]
        weights[i] = _w[0]

        # -- Insertion -------------------------------------------------- #
        arr_full[slices] = _data
        mask_full[slices] = _mask

        # process = psutil.Process(os.getpid())
        # print("3: ", process.memory_info().rss/1.e+9)  # in bytes
        del ccd, _data, _mask, _data_fake
        # process = psutil.Process(os.getpid())
        # print("4: ", process.memory_info().rss/1.e+9)  # in bytes

    if verbose:
        print("All FITS loaded, rejection & combination starts", end='... ')

    # ----------------------------------------------------------------------- #

    # == Combine with rejection! ============================================ #
    comb, std, mask_rej, mask_thresh, low, upp, nit, rejcode = ndcombine(
        arr=arr_full,
        mask=mask_full,
        copy=False,  # No need to retain arr_full.
        combine=combine,
        reject=reject,
        scale=scales,    # it is scales , NOT scale , as it was updated above.
        zero=zeros,      # it is zeros  , NOT zero  , as it was updated above.
        weight=weights,  # it is weights, NOT weight, as it was updated above.
        zero_to_0th=zero_to_0th,
        scale_to_0th=scale_to_0th,
        scale_kw=scale_kw,
        zero_kw=zero_kw,
        thresholds=thresholds,
        n_minmax=n_minmax,
        nkeep=nkeep,
        maxrej=maxrej,
        cenfunc=cenfunc,
        sigma=sigma,
        maxiters=maxiters,
        ddof=ddof,
        rdnoise=rds,  # it is rds, not rdnoise, as it was updated above.
        gain=gns,     # it is gns, not gain   , as it was updated above.
        snoise=sns,   # it is sns, not snoise , as it was updated above.
        pclip=pclip,
        irafmode=irafmode,
        full=True
    )
    comb = comb.astype(dtype)
    std = std.astype(dtype)
    low = low.astype(dtype)
    upp = upp.astype(dtype)
    # ----------------------------------------------------------------------- #

    if verbose:
        print("Done.")
        print("Making FITS", end="... ")

    mask_total = mask_full | mask_thresh | mask_rej

    # == Update header properly ============================================= #
    # Update WCS or PHYSICAL keywords so that "lock frame wcs", etc, on
    # SAO ds9, for example, to give proper visualization:
    if use_wcs:
        # NOTE: the indexing in python is [z, y, x] order!!
        for i in range(ndim, 0, -1):
            hdr0[f"CRPIX{i}"] += offsets[0][ndim - i]

    if use_phy:
        # NOTE: the indexing in python is [z, y, x] order!!
        for i in range(ndim, 0, -1):
            hdr0[f"LTV{i}"] += offsets[0][ndim - i]

    update_hdr(hdr0, ncombine, imcmb_key=imcmb_key, imcmb_val=imcmb_val,
               offset_mode=offset_mode, offsets=offsets)

    try:
        unit = hdr0["BUNIT"]
    except (KeyError, IndexError):
        unit = 'adu'

    comb = CCDData(data=comb, header=hdr0, unit=unit)

    # == Save FITS files ==================================================== #
    if output is not None:
        comb.writeto(output, **kwargs)

    if output_std is not None:
        write2fits(std, hdr0, output_std, return_ccd=False, **kwargs)

    if output_low is not None:
        write2fits(low, hdr0, output_low, return_ccd=False, **kwargs)

    if output_upp is not None:
        write2fits(upp, hdr0, output_upp, return_ccd=False, **kwargs)

    if output_nrej is not None:  # Do this BEFORE output_mask!!
        nrej = np.count_nonzero(mask_total, axis=0).astype(int_dtype)
        write2fits(nrej, hdr0, output_nrej, return_ccd=False, **kwargs)

    if output_mask is not None:  # Do this AFTER output_nrej!!
        # FITS does not accept boolean. We need uint8.
        write2fits(mask_total.astype(np.uint8), hdr0, output_mask,
                   return_ccd=False, **kwargs)

    if output_rejcode is not None:
        write2fits(rejcode, hdr0, output_rejcode, return_ccd=False, **kwargs)

    if verbose:
        print("Done.")

    # == Return memroy... =================================================== #
    del hdr0, arr_full, mask_full

    # == Return ============================================================= #
    if full:
        if return_dict:
            return dict(
                comb=comb,
                std=std,
                mask_total=mask_total,
                mask_rej=mask_rej,
                mask_thresh=mask_thresh,
                low=low,
                upp=upp,
                nit=nit,
                rejcode=rejcode
            )
        else:
            return (comb, std, mask_total, mask_rej, mask_thresh,
                    low, upp, nit, rejcode)
    else:
        return comb


fitscombine.__doc__ = '''A helper function for ndcombine to cope with FITS files.

    {}

    Parameters
    ----------
    fpaths : list-like of path-like, optional.
        The list of file paths to be combined. These must be FITS files.
        One and only one of ``fpaths`` or ``fpattern`` must be provided.

    fpattern : str, optional.
        The `~glob` pattern for files (e.g., ``"2020*[012].fits"``).
        One and only one of ``fpaths`` or ``fpattern`` must be provided.

    mask : ndarray, optional.
        The mask of bad pixels. If given, it must satisfy
        ``mask.shape[0]`` identical to the number of images.

        .. note::
            If the user ever want to use masking, it's more convenient
            to use ``'MASK'`` extension to the FITS files or replace
            bad pixel to very large or small numbers and use
            ``thresholds``.

    ext : int, optional.
        The extension to be used in loading the FITS files.

    {}

    {}

    imcmb_key : str
        The thing to add as ``IMCMBnnn`` in the output FITS file header.
        If ``"$I"``, following the default of IRAF, the file's name will
        be added. Otherwise, it should be a header keyword. If the key
        does not exist in ``nnn``-th file, a null string will be added.
        If a null string (``imcmb_key=""``), it does not set the
        ``IMCMBnnn`` keywords nor deletes any existing keyword.

        .. warning::
            If more than 999 files are combined, only the first 999
            files will be recorded in the header.

    exposure_key : str, optional.
        The header keyword which contains the information about the
        exposure time of each FITS file. This is used only if scaling is
        done for exposure time (see ``scale``).

    irafmode : bool, optional.
        Whether to use IRAF-like pixel restoration scheme.

    output : path-like, optional
        The path to the final combined FITS file. It has dtype of
        ``dtype`` and dimension identical to each input image.
        Optional keyword arguments for ``fits.writeto()`` can be
        provided as ``**kwargs``.

    output_xxx : path-like, optional
        The output path to the mask, number of rejected pixels at each
        position, final ``nanstd(ddof=ddof)`` result,
        lower and upper bounds for rejection, and the integer codes for
        the rejection algorithm (see ``mask_total``, ``mask_rej``,
        ``sigma``, ``low``, ``upp``, and ``rejcode`` in Returns.)

    return_dict : bool, optional.
        Whether to return the results as dict (works only if
        ``full=True``).

    Returns
    -------
    Returns the followings depending on ``full`` and ``return_dict``.

    comb : `astropy.nddata.CCDData` (dtype ``dtype``)
        The combined data.

    {}

    {}
    '''.format(docstrings.NDCOMB_NOT_IMPLEMENTED(indent=4),
               docstrings.OFFSETS_LONG(indent=4),
               docstrings.NDCOMB_PARAMETERS_COMMON(indent=4),
               docstrings.NDCOMB_RETURNS_COMMON(indent=4),
               docstrings.IMCOMBINE_LINK(indent=4))


# --------------------------------------------------------------------------- #
def ndcombine(
        arr, mask=None, copy=True,
        blank=np.nan,
        offsets=None,
        thresholds=[-np.inf, np.inf],
        zero=None, scale=None, weight=None,
        zero_kw={'cenfunc': 'median', 'stdfunc': 'std', 'std_ddof': 1},
        scale_kw={'cenfunc': 'median', 'stdfunc': 'std', 'std_ddof': 1},
        zero_to_0th=True, scale_to_0th=True,
        zero_section=None, scale_section=None,
        reject=None,
        cenfunc='median',
        sigma=[3., 3.], maxiters=3, ddof=1, nkeep=1, maxrej=None,
        n_minmax=[1, 1],
        rdnoise=0., gain=1., snoise=0.,
        pclip=-0.5,
        combine='average',
        dtype='float32',
        memlimit=2.5e+9,
        irafmode=True,
        verbose=False,
        full=False,
):
    if copy:
        arr = arr.copy()

    if np.array(arr).ndim == 1:
        raise ValueError("1-D array combination is not supported!")

    _mask = _set_mask(arr, mask)  # _mask = propagated through this function.
    sigma_lower, sigma_upper = _set_sigma(sigma)
    nkeep, maxrej = _set_keeprej(arr, nkeep, maxrej, axis=0)
    cenfunc = _set_cenfunc(cenfunc)
    reject = _set_reject_name(reject)
    maxiters = int(maxiters)
    ddof = int(ddof)

    combfunc = _set_combfunc(combine, nameonly=False, nan=True)

    # == 01 - Thresholding + Initial masking ================================ #
    # Updating mask: _mask = _mask | mask_thresh
    mask_thresh = _set_thresh_mask(
        arr=arr,
        mask=_mask,
        thresholds=thresholds,
        update_mask=True
    )

    # if safemode:
    #     # Backup the pixels which are rejected by thresholding and
    #     # initial mask for future restoration (see below) for debugging
    #     # purpose.
    #     backup_thresh = arr[mask_thresh]
    #     backup_thresh_inmask = arr[_mask]

    arr[_mask] = np.nan
    # ----------------------------------------------------------------------- #

    # == 02 - Calculate zero, scale, weights ================================ #
    # This should be done before rejection but after threshold masking..
    zeros, scales, weights = get_zsw(
        arr=arr,
        zero=zero,
        scale=scale,
        weight=weight,
        zero_kw=zero_kw,
        scale_kw=scale_kw,
        zero_to_0th=zero_to_0th,
        scale_to_0th=scale_to_0th,
        zero_section=zero_section,
        scale_section=scale_section
    )
    arr = do_zs(arr, zeros=zeros, scales=scales)
    # ----------------------------------------------------------------------- #

    # == 02 - Rejection ===================================================== #
    if isinstance(reject, str):
        if reject == 'sigclip':
            _mask_rej, low, upp, nit, rejcode = sigclip_mask(
                arr,
                mask=_mask,
                sigma_lower=sigma_lower,
                sigma_upper=sigma_upper,
                maxiters=maxiters,
                ddof=ddof,
                nkeep=nkeep,
                maxrej=maxrej,
                cenfunc=cenfunc,
                axis=0,
                irafmode=irafmode,
                full=True
            )
            # _mask is a subset of _mask_rej, so to extract pixels which
            # are masked PURELY due to the rejection is:
            mask_rej = _mask_rej ^ _mask
        elif reject == 'minmax':
            pass
        elif reject == 'ccdclip':
            _mask_rej, low, upp, nit, rejcode = ccdclip_mask(
                arr,
                mask=_mask,
                sigma_lower=sigma_lower,
                sigma_upper=sigma_upper,
                scale_ref=np.mean(scales),
                zero_ref=np.mean(zeros),
                maxiters=maxiters,
                ddof=ddof,
                nkeep=nkeep,
                maxrej=maxrej,
                cenfunc=cenfunc,
                axis=0,
                gain=gain,
                rdnoise=rdnoise,
                snoise=snoise,
                irafmode=irafmode,
                full=True
            )
            # _mask is a subset of _mask_rej, so to extract pixels which
            # are masked PURELY due to the rejection is:
            mask_rej = _mask_rej ^ _mask
        elif reject == 'pclip':
            pass
        else:
            raise ValueError("reject not understood.")
    elif reject is None:
        mask_rej = _set_mask(arr, None)
        low = bn.nanmin(arr, axis=0)
        upp = bn.nanmax(arr, axis=0)
        nit = None
        rejcode = None
    else:
        raise ValueError("reject not understood.")

    _mask |= mask_rej

    # ----------------------------------------------------------------------- #

    # TODO: add "grow" rejection here?

    # == 03 - combine ======================================================= #
    # Replace rejected / masked pixel to NaN and backup for debugging purpose.
    # This is done to reduce memory (instead of doing _arr = arr.copy())
    # backup_nan = arr[_mask]
    arr[_mask] = np.nan

    # Combine and calc sigma
    comb = combfunc(arr, axis=0)
    if full:
        std = bn.nanstd(arr, ddof=ddof, axis=0)

    # Restore NaN-replaced pixels of arr for debugging purpose.
    # arr[_mask] = backup_nan
    # arr[mask_thresh] = backup_thresh_inmask

    if full:
        return comb, std, mask_rej, mask_thresh, low, upp, nit, rejcode
    else:
        return comb


ndcombine.__doc__ = ''' Combines the given arr assuming no additional offsets.

    {}
    #. offsets is not implemented to ndcombine (only to fitscombine).

    Parameters
    ----------
    arr : ndarray
        The array to be combined along axis 0.

    mask : ndarray, optional.
        The mask of bad pixels. If given, it must satisfy
        ``mask.shape[0]`` identical to the number of images.

    copy : bool, optional.
        Whether to copy the input array. Set to `True` if you want to
        keep the original array unchanged. Otherwise, the original array
        may be destroyed.

    {}

    {}

    Returns
    -------
    comb : ndarray
        The combined array.

    {}

    {}
    '''.format(docstrings.NDCOMB_NOT_IMPLEMENTED(indent=4),
               docstrings.OFFSETS_SHORT(indent=4),
               docstrings.NDCOMB_PARAMETERS_COMMON(indent=4),
               docstrings.NDCOMB_RETURNS_COMMON(indent=4),
               docstrings.IMCOMBINE_LINK(indent=4)
               )
