import glob
from pathlib import Path

import bottleneck as bn
import numpy as np
from astropy.io import fits
from numpy.lib.arraysetops import isin

from .reject import sigclip_mask
from .util import (_get_combine_shape, _set_cenfunc, _set_combfunc,
                   _set_int_dtype, _set_keeprej, _set_mask, _set_reject_name,
                   _set_sigma, _set_thresh_mask, do_zs, get_zsw, update_hdr,
                   write2fits)

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

'''

# TODO:
add blank option
add logfile
add statsec with input, output, overlap
add scale, zero, weight
add scale_sample, zero_sample
add mode for scale, zero, weight?
add memlimit behaviour
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
        exposure_key="EXPTIME",
        scale_sample=None, zero_sample=None,
        reject=None,
        cenfunc='median',
        sigma=[3., 3.], maxiters=1, ddof=1, nkeep=1, maxrej=None,
        n_minmax=[1, 1],
        rdnoise=0., gain=1., snoise=0.,
        pclip=-0.5,
        logfile=None,
        combine='average',
        dtype='float32',
        satlevel=65535, irafmode=False,
        memlimit=2.5e+9,
        verbose=False,
        full=False,
        imcmb_key='$I',
        output=None, output_mask=None, output_nrej=None,
        output_sigma=None, output_low=None, output_upp=None,
        output_rejcode=None,
        **kwargs
):
    '''A helper function for ndcombine to cope with FITS files.

    .. warning::
        Few functionalities are not implemented yet:

            #. ``blank`` option
            #. ``logfile``
            #. ``statsec`` with input, output, overlap
            #. ``weight``
            #. ``scale_sample``, ``zero_sample``
            #. ``"mode"`` for ``scale``, ``zero``, ``weight``
            #. ``memlimit`` behaviour

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

    offsets : str or (n, m)-d array
        If array, it must have shape such that ``n`` is the number of
        images and ``m`` is the dimension of the images (if ``m=3``,
        offsets in x, y, z, ... order, not pythonic order), and it is
        directly regarded as the **raw offsets**. If ``str``, the raw
        offsets are obtained by the followings:
          - ``CRPIX`` values in the header if ``"wcs"|"world"``
          - ``LTV`` values in the header if ``"physical"|"phys"|"phy"``

        .. note::
            The physical coordinate system is defined by the IRAF-like
            ``LTM``/``LTV`` keywords define the offsets. Currently,
            only the cases when ``LTMi_j`` is 0 or 1 can be managed.
            Otherwise, we need scaling and it is not supported now.

        For both wcs or physical cases, the raw offsets for *each* frame
        is nothing but an ``m``-D tuple consists of
        ``offset_raw[i] = CRPIX{m-i}`` or ``LTV{m-i}[_{m-i}]``.
        The reason to subtract ``i`` is because python has ``z, y, x``
        order of indexing while WCS information is in ``x, y, z`` order.
        If it is a ``j``-th image, ``offsets[j, :] = offset_raw``, and
        ``offsets`` has shape of ``(n, m)``.

        This raw ``offsets`` are then modified such that the minimum
        offsets in each axis becomes zero (in pythonic way,
        ``np.max(offsets, axis=0) - offsets``). The offsets are used
        to determine the final output image's shape.

        .. note::
            Though IRAF imcombine says it calculates offsets from the
            0-th image center if ``offsets="wcs"``, it seems it acutally
            uses ``CRPIX`` from the header... I couldn't find how IRAF
            does offset calculation for WCS, it's not reproducible using
            rounding... Even using WCS info correctly, it's not
            reproducible.
            Very curious. But that mismatch is at most 2 pixels, and
            mostly 0 pixel, so let me not bothered by it.

    thresholds : 2-float list-like, optional.
        The thresholds ``(lower, upper)`` applied to all images before
        any rejection/combination. Default is no thresholding,
        ``(-np.inf, +np.inf)``. One possible usage is to replace bad
        pixel to very large or small numbers and use this thresholding.

    zero : str or 1-d array
        The *zero* value to subtract from each image *after*
        thresholding, but *before* scaling/offset
        shifting/rejection/combination. If an array, it is directly
        subtracted from each image, (so it must
        have size identical to the number of images). If ``str``, the
        zero-level is:
          - **Average** if ``'avg'|'average'|'mean'``
          - **Median** if ``'med'|'medi'|'median'``
          - **Sigma-clipped average** if
            ``'avg_sc'|'average_sc'|'mean_sc'``
          - **Sigma-clipped median** if
            ``'med_sc'|'medi_sc'|'median_sc'``
        For options for sigma-clipped statistics, see ``zero_kw``.

        .. note::
            By using ``zero="med_sc"``, the user can crudely subtract
            sky value from each frame before combining.

    scale : str or 1-d array
        The way to scale each image *after* thresholding/zeroing, but
        *before* offset shifting/rejection/combination. If an array, it
        is directly understood as the **raw scales**, and it must have
        size identical to the number of images. If ``str``, the raw
        scale is:
          - **Exposure time** (``exposure_key`` in header of each FITS)
            if ``'exp'|'expos'|'exposure'|'exptime'``
          - **Average** if ``'avg'|'average'|'mean'``
          - **Median** if ``'med'|'medi'|'median'``
          - **Sigma-clipped average** if
            ``'avg_sc'|'average_sc'|'mean_sc'``
          - **Sigma-clipped median** if
            ``'med_sc'|'medi_sc'|'median_sc'``
        The true scale is obtained by ``scales / scales[0]`` if
        ``scale_to_0th`` is `True`, following IRAF's convention.
        Otherwise, the absolute value from the raw scale will be used.
        For options for sigma-clipped statistics, see ``scale_kw``.

        .. note::
            Using ``scale="avg_sc", scale_to_0th=False`` is useful for
            flat combining.

    zero_to_0th : bool, optional.
        Whether to re-base the zero values such that all images have
        identical zero values as that of the 0-th image (in python,
        ``zero - zero[0]``). This is the behavior of IRAF, so
        ``zero_to_0th`` is `True` by default.

    scale_to_0th : bool, optional.
        Whether to re-scale the scales such that ``scale[0]`` is unity.
        This is the behavior of IRAF, so ``scale_to_0th`` is `True`
        by default.

    zero_kw, scale_kw : dict
        Used only if ``scale`` or ``zero`` are sigma-clipped mean,
        median, etc (ending with ``_sc`` such as ``median_sc``,
        ``avg_sc``). The keyword arguments for
        `astropy.stats.sigma_clipped_stats`. By default,
        ``std_ddof=1`` (note that `~astropy.stats.sigma_clipped_stats`
        has default ``std_ddof=0``.)

        .. note::
            If ``axis`` is specified, it will be ignored.

    exposure_key : str, optional.
        The header keyword which contains the information about the
        exposure time of each FITS file. This is used only if scaling is
        done for exposure time (see ``scale``).

    sigma : 2-float list-like, optional.
        The sigma-factors to be used for sigma-clip rejeciton in
        ``(sigma_lower, sigma_upper)``. Defaults to ``(3, 3)``, which
        means 3-sigma clipping from the "sigma" values determined by the
        method specified by ``reject``. If a single float, it will be
        used for both the lower and upper values.

    maxiters : int, optional.
        The maximum number of iterations to do the rejection (for
        sigma-clipping). It is silently converted to ``int`` if it is
        not.

    ddof : int, optional.
        The delta-degrees of freedom (see `numpy.std`). It is silently
        converted to ``int`` if it is not.

    nkeep : float or int, optional.
        The minimum number of pixels that should be left after
        rejection. If ``nkeep < 1``, it is regarded as fraction of the
        total number of pixels along the axis to combine. This
        corresponds to *positive* ``nkeep`` parameter of IRAF IMCOMBINE.
        If number of remaining non-nan value is fewer than ``nkeep``,
        the masks at that position will be reverted to the previous
        iteration, and rejection code will be added by number 4.

    maxrej : float or int, optional.
        The maximum number of pixels that can be rejected during the
        rejection. If ``maxrej < 1``, it is regarded as fraction of the
        total number of pixels along the axis to combine. This
        corresponds to *negative* ``nkeep`` parameter of IRAF IMCOMBINE.
        In IRAF, only one of ``nkeep`` and ``maxrej`` can be set.
        If number of rejected pixels at a position exceeds ``maxrej``,
        the masks at that position will be reverted to the previous
        iteration, and rejection code will be added by number 8.

    cenfunc : str, optional.
        The centering function to be used in rejection algorithm.

          - median if  ``'med'|'medi'|'median'``
          - average if ``'avg'|'average'|'mean'``
          - lower median if ``'lmed'|'lmd'|'lmedian'``

        For lower median, see note in ``combine``.

    n_minmax : 2-float or 2-int list-like, optional.
        The number of low and high pixels to be rejected by the "minmax"
        algorithm. These numbers are converted to fractions of the total
        number of input images so that if no rejections have taken place
        the specified number of pixels are rejected while if pixels have
        been rejected by masking, thresholding, or non-overlap, then the
        fraction of the remaining pixels, truncated to an integer, is
        used.

    rdnoise, gain, snoise : float, optional.
        The readnoise of the detector in the unit of electrons, electron
        gain of the detector in the unit of elctrons/DN (or
        electrons/ADU), and sensitivity noise as a fraction. Used only
        if ``reject="ccdclip"`` and/or ``combine="nmodel"``.

        The variance of a single pixel in an image when these are used,

        .. math::
            V_\mathrm{DN}
            = ( \mathtt{rdnoise}/\mathtt{gain} )^2
            + \mathrm{DN}/\mathtt{gain}
            + ( \mathtt{snoise} * \mathrm{DN} )^2

        .. math::
            V_\mathrm{electron}
            = (\mathtt{rdnoise})^2
            + (\mathtt{gain} * \mathrm{DN})^2
            + (\mathtt{snoise} * \mathtt{gain} * \mathrm{DN})^2

    pclip : float, optional.
        The parameter for ``reject="pclip"``. If ``abs(pclip) >= 1``,
        then it specifies a number of pixels above or below the median
        to use for computing the clipping sigma. If ``abs(pclip) < 1``,
        then it specifies the fraction of the pixels above or below the
        median to use. A positive value selects a point above the median
        and a negative value selects a point below the median. The
        default of ``-0.5`` selects approximately the quartile point.
        Better to use negative value to avoid cosmic-ray contamination.

    combine: str, optional.
        The function to be used for the final combining after
        thresholding, zeroing, scaling, rejection, and offset shifting.

          - median if  ``'med'|'medi'|'median'``
          - average if ``'avg'|'average'|'mean'``
          - lower median if ``'lmed'|'lmd'|'lmedian'``

        .. note::
            The lower median means the median which takes the lower value
            when even number of data is left. This is suggested to be robust
            against cosmic-ray hit according to IRAF IMCOMBINE manual.
            Currently there is no lmedian-alternative in bottleneck or
            numpy, so a custom-made version is used (in `numpy_util.py`),
            which is nothing but a simple modification to the original
            numpy source codes, and this is much slower than
            bottleneck's median. I think it must be re-implemented in
            the future.

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

    Returns
    -------
    comb : `astropy.io.fits.PrimaryHDU`
        The combined FITS file.

    sigma : ndarray
        The sigma map

    mask_total : ndarray (dtype bool)
        The full mask, ``N+1``-D. Identical to original FITS files'
        masks propagated with ``| mask_rej | mask_thresh`` below. The
        total number of rejected pixels at each position can be obtained
        by ``np.sum(mask_total, axis=0)``.

    mask_rej, mask_thresh : ndarray(dtype bool)
        The masks (``N``-D) from the rejection process and thresholding
        process (``thresholds``). Threshold is done prior to any
        rejection or scaling/zeroing. Number of rejected pixels at each
        position for each process can be obtained by, e.g.,
        ``nrej = np.sum(mask_rej, axis=0)``. Note that ``mask_rej``
        consumes less memory than ``nrej``.

    low, upp : ndarray (dtype ``dtype``)
        The lower and upper bounds (``N``-D) to reject pixel values at
        each position (``(data < low) | (upp < data)`` are removed).

    nit : ndarray (dtype uint8)
        The number of iterations (``N``-D) used in rejection process. I
        cannot think of iterations larger than 100, so set the dtype to
        ``uint8`` to reduce memory and filesize.

    rejcode : ndarray (dtype uint8)
        The exit code from rejection (``N``-D). See each rejection's
        docstring.
    '''
    if verbose:
        print("Organizing", end='... ')
    given_fpaths = fpaths is not None
    given_fpattern = fpattern is not None

    if given_fpaths + given_fpattern != 1:
        raise ValueError("Give one and only one of fpaths/fpattern.")

    if given_fpattern:
        fpaths = list(glob.glob(fpattern))
        fpaths.sort()

    fpaths = list(fpaths)  # convert to list just in case...
    ncombine = len(fpaths)
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
            scale = []

    # == organize offsets =================================================== #
    # TODO: if offsets is None and ``fsize_tot`` << memlimit, why not
    # just load all data here?
    # initialize
    use_wcs, use_phy = False, False
    if isinstance(offsets, str):
        if offsets.lower() in ['world', 'wcs']:
            # w_ref = WCS(hdr0)
            # cen_ref = np.array([hdr0[f'NAXIS{i+1}']/2 for i in range(ndim)])
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
        hdul = fits.open(fpath, ext=ext)
        hdr = hdul[ext].header
        if imcmb_key != '':
            if imcmb_key == "$I":
                imcmb_val.append(fpath.name)
            else:
                try:
                    imcmb_val.append(hdr[imcmb_key])
                except KeyError:
                    imcmb_val.append('')

        if extract_exptime:
            scale.append(float(hdr[exposure_key]))

        if hdr['NAXIS'] != ndim:
            raise ValueError(
                "All FITS files must have the identical dimensionality, "
                + "though they can have different sizes."
            )

        # Update offsets if WCS or Physical should be used
        if use_wcs:
            # NOTE: the indexing in python is [z, y, x] order!!
            offsets[i, ] = [hdr[f'CRPIX{i}'] for i in range(ndim, 0, -1)]
            '''
            # Code if using WCS, which may be much slower (but accurate?)
            # Find the center's pixel position in w_ref, in nearest int
            from astropy.wcs import WCS
            w = WCS(hdr)
            cen = [hdr[f'NAXIS{i+1}']/2 for i in range(ndim)]
            cen_coo = w.all_pix2world(*cen, 0)
            cen = np.around(w_ref.all_world2pix(*cen_coo, 0)).astype(int)
            offsets[i, ] = cen_ref - cen
            '''
        elif use_phy:
            # NOTE: the indexing in python is [z, y, x] order!!
            offsets[i, ] = [hdr[f'LTV{i}'] for i in range(ndim, 0, -1)]

        # NOTE: the indexing in python is [z, y, x] order!!
        sizes[i, ] = [int(hdr[f'NAXIS{i}']) for i in range(ndim, 0, -1)]
    # ----------------------------------------------------------------------- #

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
        # -- Set slice ------------------------------------------------------ #
        slices = [i]
        # offset & size at each j-th dimension axis
        for offset_j, size_j in zip(_offset, _size):
            slices.append(slice(offset_j, offset_j + size_j, None))

        # -- Set mask ------------------------------------------------------- #
        with fits.open(_fpath) as hdul:
            _data = hdul[ext].data
            try:  # load MASK from FITS file if exists
                _mask = hdul["MASK"].data.astype('bool')
            except KeyError:
                _mask = np.zeros(hdul[ext].data.shape, dtype=bool)

            if mask is not None:
                _mask |= mask[i, ]

            # -- zero and scale --------------------------------------------- #
            # better to calculate here than from full array, as the
            # latter may contain too many NaNs due to offest shifting.
            _z, _s, _w = get_zsw(
                arr=_data[None, :],  # make a fake (N+1)-D array
                zero=zero,
                scale=scale,
                weight=weight,
                zero_kw=zero_kw,
                scale_kw=scale_kw,
                zero_to_0th=False,  # to retain original zero
                scale_to_0th=False  # to retain original scale
            )
            zeros[i] = _z[0]
            scales[i] = _s[0]
            weights[i] = _w[0]

            # -- Insertion -------------------------------------------------- #
            arr_full[slices] = _data
            mask_full[slices] = _mask

            del hdul[ext].data

    if verbose:
        print("All FITS loaded, rejection & combination starts", end='... ')
    # ----------------------------------------------------------------------- #

    # == Combine with rejection! ============================================ #
    comb, sigma, mask_rej, mask_thresh, low, upp, nit, rejcode = ndcombine(
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
        statsec=statsec,
        thresholds=thresholds,
        n_minmax=n_minmax,
        nkeep=nkeep,
        maxrej=maxrej,
        cenfunc=cenfunc,
        sigma=sigma,
        maxiters=maxiters,
        ddof=ddof,
        rdnoise=rdnoise,
        gain=gain,
        snoise=snoise,
        pclip=pclip,
        satlevel=satlevel,
        irafmode=irafmode,
        full=True
    )
    comb = comb.astype(dtype)
    sigma = sigma.astype(dtype)
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
    comb = fits.PrimaryHDU(data=comb, header=hdr0)

    # == Save FITS files ==================================================== #
    if output is not None:
        comb.writeto(output, **kwargs)

    if output_sigma is not None:
        write2fits(sigma, hdr0, output_sigma, return_hdu=False, **kwargs)

    if output_low is not None:
        write2fits(low, hdr0, output_low, return_hdu=False, **kwargs)

    if output_upp is not None:
        write2fits(upp, hdr0, output_upp, return_hdu=False, **kwargs)

    if output_nrej is not None:  # Do this BEFORE output_mask!!
        nrej = np.sum(mask_total, axis=0).astype(int_dtype)
        write2fits(nrej, hdr0, output_nrej, return_hdu=False, **kwargs)

    if output_mask is not None:  # Do this AFTER output_nrej!!
        # FITS does not accept boolean. We need uint8.
        write2fits(mask_total.astype(np.uint8), hdr0, output_mask,
                   return_hdu=False, **kwargs)

    if output_rejcode is not None:
        write2fits(rejcode, hdr0, output_rejcode, return_hdu=False, **kwargs)

    if verbose:
        print("Done.")

    # == Return ============================================================= #
    if full:
        return (comb, sigma, mask_total, mask_rej, mask_thresh,
                low, upp, nit, rejcode)
    else:
        return comb


# --------------------------------------------------------------------------- #
def ndcombine(
        arr, mask=None, copy=True,
        blank=np.nan,
        offsets=None,
        thresholds=[-np.inf, np.inf],
        zero=None, scale=None, weight=None, statsec=None,
        zero_kw={'cenfunc': 'median', 'stdfunc': 'std', 'std_ddof': 1},
        scale_kw={'cenfunc': 'median', 'stdfunc': 'std', 'std_ddof': 1},
        zero_to_0th=True, scale_to_0th=True,
        scale_sample=None, zero_sample=None,
        reject=None,
        cenfunc='median',
        sigma=[3., 3.], maxiters=3, ddof=1, nkeep=1, maxrej=None,
        n_minmax=[1, 1],
        rdnoise=0., gain=1., snoise=0.,
        pclip=-0.5,
        combine='average',
        dtype='float32',
        memlimit=2.5e+9,
        satlevel=65535, irafmode=False,
        verbose=False,
        full=False
):
    ''' Combines the given arr assuming no additional offsets.

    .. warning::
        Few functionalities are not implemented yet:

            #. ``blank`` option
            #. ``logfile``
            #. ``statsec`` with input, output, overlap
            #. ``weight``
            #. ``scale_sample``, ``zero_sample``
            #. ``"mode"`` for ``scale``, ``zero``, ``weight``
            #. ``memlimit`` behaviour

    Parameters
    ----------
    arr : ndarray
        The array to be combined along axis 0.

    mask : ndarray, optional.
        The mask of bad pixels. If given, it must satisfy
        ``mask.shape[0]`` identical to the number of images.

    copy : bool, optional.
        Whether to copy the input array. Set to `True` if you want to
        keep the original array unchanged. Even if it is `False`, the
        code tries to keep ``arr`` unchanged unless, e.g.,
        KeyboardInterrupt.

    offsets : (n, m)-d array
        If given, it must have shape such that ``n`` is the number of
        images and ``m`` is the dimension of the images (offsets in x,
        y, z, ... order, not pythonic order), and it is directly
        regarded as the "raw offsets".

        The raw offsets are then modified such that the minimum offsets
        in each axis becomes zero (in pythonic way,
        ``np.max(offsets, axis=0) - offsets``). The offsets are used
        to determine the final output image's shape.

    thresholds : 2-float list-like, optional.
        The thresholds ``(lower, upper)`` applied to all images before
        any rejection/combination. Default is no thresholding,
        ``(-np.inf, +np.inf)``. One possible usage is to replace bad
        pixel to very large or small numbers and use this thresholding.

    zero : str or 1-d array
        The "zero" value to subtract from each image *after*
        thresholding, but *before* scaling/offset
        shifting/rejection/combination. If an array, it is directly
        subtracted from each image, (so it must
        have size identical to the number of images). If ``str``, the
        zero-level is:
          - **Average** if ``'avg'|'average'|'mean'``
          - **Median** if ``'med'|'medi'|'median'``
          - **Sigma-clipped average** if
            ``'avg_sc'|'average_sc'|'mean_sc'``
          - **Sigma-clipped median** if
            ``'med_sc'|'medi_sc'|'median_sc'``
        For options for sigma-clipped statistics, see ``zero_kw``.

        .. note::
            By using ``zero="med_sc"``, the user can crudely subtract
            sky value from each frame before combining.

    scale : str or 1-d array
        The way to scale each image *after* thresholding/zeroing, but
        *before* offset shifting/rejection/combination. If an array, it
        is directly understood as the **raw scales**, and it must have
        size identical to the number of images. If ``str``, the raw
        scale is:
          - **Exposure time** (``exposure_key`` in header of each FITS)
            if ``'exp'|'expos'|'exposure'|'exptime'``
          - **Average** if ``'avg'|'average'|'mean'``
          - **Median** if ``'med'|'medi'|'median'``
          - **Sigma-clipped average** if
            ``'avg_sc'|'average_sc'|'mean_sc'``
          - **Sigma-clipped median** if
            ``'med_sc'|'medi_sc'|'median_sc'``
        The true scale is obtained by ``scales / scales[0]`` if
        ``scale_to_0th`` is `True`, following IRAF's convention.
        Otherwise, the absolute value from the raw scale will be used.
        For options for sigma-clipped statistics, see ``scale_kw``.

        .. note::
            Using ``scale="avg_sc", scale_to_0th=False`` is useful for
            flat combining.

    zero_to_0th : bool, optional.
        Whether to re-base the zero values such that all images have
        identical zero values as that of the 0-th image (in python,
        ``zero - zero[0]``). This is the behavior of IRAF, so
        ``zero_to_0th`` is `True` by default.

    scale_to_0th : bool, optional.
        Whether to re-scale the scales such that ``scale[0]`` is unity.
        This is the behavior of IRAF, so ``scale_to_0th`` is `True`
        by default.

    zero_kw, scale_kw : dict
        Used only if ``scale`` or ``zero`` are sigma-clipped mean,
        median, etc (ending with ``_sc`` such as ``median_sc``,
        ``avg_sc``). The keyword arguments for astropy's
        `astropy.stats.sigma_clipped_stats`. By default,
        ``std_ddof=1``, which is different from that of original
        ``sigma_clipped_stats``.

        .. note::
            If ``axis`` is specified, it will be ignored.

    sigma : float, 2-float list-like, optional.
        The sigma-factors to be used for sigma-clip rejeciton in
        ``(sigma_lower, sigma_upper)``. Defaults to ``(3, 3)``, which
        means 3-sigma clipping from the "sigma" values determined by the
        method specified by ``reject``. If a single float, it will be
        used for both the lower and upper values.

    maxiters : int, optional.
        The maximum number of iterations to do the rejection (for
        sigma-clipping). It is silently converted to ``int`` if it is
        not.

    ddof : int, optional.
        The delta-degrees of freedom (see `numpy.std`). It is silently
        converted to ``int`` if it is not.

    nkeep : float or int, optional.
        The minimum number of pixels that should be left after
        rejection. If ``nkeep < 1``, it is regarded as fraction of the
        total number of pixels along the axis to combine. This
        corresponds to *positive* ``nkeep`` parameter of IRAF IMCOMBINE.
        If number of remaining non-nan value is fewer than ``nkeep``,
        the masks at that position will be reverted to the previous
        iteration, and rejection code will be added by number 4.

    maxrej : float or int, optional.
        The maximum number of pixels that can be rejected during the
        rejection. If ``maxrej < 1``, it is regarded as fraction of the
        total number of pixels along the axis to combine. This
        corresponds to *negative* ``nkeep`` parameter of IRAF IMCOMBINE.
        In IRAF, only one of ``nkeep`` and ``maxrej`` can be set.
        If number of rejected pixels at a position exceeds ``maxrej``,
        the masks at that position will be reverted to the previous
        iteration, and rejection code will be added by number 8.

    cenfunc : str, optional.
        The centering function to be used in rejection algorithm.

          * median if  ``'med'|'medi'|'median'``
          * average if ``'avg'|'average'|'mean'``
          * lower median if ``'lmed'|'lmd'|'lmedian'``

        For lower median, see note in ``combine``.

    n_minmax : 2-float or 2-int list-like, optional.
        The number of low and high pixels to be rejected by the "minmax"
        algorithm. These numbers are converted to fractions of the total
        number of input images so that if no rejections have taken place
        the specified number of pixels are rejected while if pixels have
        been rejected by masking, thresholding, or non-overlap, then the
        fraction of the remaining pixels, truncated to an integer, is
        used.

    rdnoise, gain, snoise : float, optional.
        The readnoise of the detector in the unit of electrons, electron
        gain of the detector in the unit of elctrons/DN (or
        electrons/ADU), and sensitivity noise as a fraction. Used only
        if ``reject="ccdclip"`` and/or ``combine="nmodel"``.

        The variance of a single pixel in an image when these are used,

        .. math::
            V_\mathrm{DN}
            = ( \mathtt{rdnoise}/\mathtt{gain} )^2
            + \mathrm{DN}/\mathtt{gain}
            + ( \mathtt{snoise} * \mathrm{DN} )^2

        .. math::
            V_\mathrm{electron}
            = (\mathtt{rdnoise})^2
            + (\mathtt{gain} * \mathrm{DN})^2
            + (\mathtt{snoise} * \mathtt{gain} * \mathrm{DN})^2

    pclip : float, optional.
        The parameter for ``reject="pclip"``. If ``abs(pclip) >= 1``,
        then it specifies a number of pixels above or below the median
        to use for computing the clipping sigma. If ``abs(pclip) < 1``,
        then it specifies the fraction of the pixels above or below the
        median to use. A positive value selects a point above the median
        and a negative value selects a point below the median. The
        default of ``-0.5`` selects approximately the quartile point.
        Better to use negative value to avoid cosmic-ray contamination.

    combine: str, optional.
        The function to be used for the final combining after
        thresholding, zeroing, scaling, rejection, and offset shifting.

          - median if  ``'med'|'medi'|'median'``
          - average if ``'avg'|'average'|'mean'``
          - lower median if ``'lmed'|'lmd'|'lmedian'``

        .. note::
            The lower median means the median which takes the lower value
            when even number of data is left. This is suggested to be robust
            against cosmic-ray hit according to IRAF IMCOMBINE manual.
            Currently there is no lmedian-alternative in bottleneck or
            numpy, so a custom-made version is used (in `numpy_util.py`),
            which is nothing but a simple modification to the original
            numpy source codes, and this is much slower than
            bottleneck's median. I think it must be re-implemented in
            the future.

    Returns
    -------
    comb : `astropy.io.fits.PrimaryHDU`
        The combined FITS file.

    sigma : ndarray
        The sigma map

    mask_total : ndarray (dtype bool)
        The full mask, ``N+1``-D. Identical to original FITS files'
        masks propagated with ``| mask_rej | mask_thresh`` below. The
        total number of rejected pixels at each position can be obtained
        by ``np.sum(mask_total, axis=0)``.

    mask_rej, mask_thresh : ndarray(dtype bool)
        The masks (``N``-D) from the rejection process and thresholding
        process (``thresholds``). Threshold is done prior to any
        rejection or scaling/zeroing. Number of rejected pixels at each
        position for each process can be obtained by, e.g.,
        ``nrej = np.sum(mask_rej, axis=0)``. Note that ``mask_rej``
        consumes less memory than ``nrej``.

    low, upp : ndarray (dtype ``dtype``)
        The lower and upper bounds (``N``-D) to reject pixel values at
        each position (``(data < low) | (upp < data)`` are removed).

    nit : ndarray (dtype uint8)
        The number of iterations (``N``-D) used in rejection process. I
        cannot think of iterations larger than 100, so set the dtype to
        ``uint8`` to reduce memory and filesize.

    rejcode : ndarray (dtype uint8)
        The exit code from rejection (``N``-D). See each rejection's
        docstring.
    '''
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

    ndim = arr.ndim
    ncombine = arr.shape[0]

    if offsets is None:
        offsets = np.zeros((ncombine, ndim))
    else:
        if offsets.shape[0] != ncombine:
            raise ValueError("offset.shape[0] must be num(images)")

    combfunc = _set_combfunc(combine, nameonly=False, nan=True)

    # == 01 - Thresholding + Initial masking ================================ #
    mask_thresh = _set_thresh_mask(
        arr=arr,
        mask=_mask,
        thresholds=thresholds,
        update_mask=True
    )

    # Backup the pixels which are rejected by thresholding for future
    # restoration (see below) for debugging purpose.
    backup_thresh = arr[mask_thresh]
    arr[_mask] = np.nan
    # ----------------------------------------------------------------------- #

    # == 02 - Calculate zero, scale, weights ================================ #
    # This should be done before rejection but after threshold masking..
    # If it were done before threshold masking, it must have been much easier.
    zeros, scales, weights = get_zsw(
        arr=arr,
        zero=zero,
        scale=scale,
        weight=weight,
        zero_kw=zero_kw,
        scale_kw=scale_kw,
        zero_to_0th=zero_to_0th,
        scale_to_0th=scale_to_0th
    )
    arr = do_zs(arr, zeros=zeros, scales=scales)
    # ----------------------------------------------------------------------- #

    # == 02 - Rejection ===================================================== #
    if reject == 'sigclip':
        mask_rej, low, upp, nit, rejcode = sigclip_mask(
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
            satlevel=satlevel,
            irafmode=irafmode,
            full=True
        )
    elif reject == 'minmax':
        pass
    elif reject == 'ccdclip':
        pass
    elif reject == 'pclip':
        pass
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
    backup_nan = arr[_mask]
    arr[_mask] = np.nan

    # Combine and restore NaN-replaced pixels of arr for debugging purpose.
    comb = combfunc(arr, axis=0)
    if full:
        sigma = bn.nanstd(arr, axis=0)

    arr[_mask] = backup_nan
    arr[mask_thresh] = backup_thresh

    if full:
        return comb, sigma, mask_rej, mask_thresh, low, upp, nit, rejcode
    else:
        return comb


'''
    arr = np.array(arr)
    ncombine = arr.shape[0]
    ndim = arr.ndim
    if ndim < 2:
        raise ValueError("arr must have ndim > 1!")
    memused = arr.size*arr.itemsize
    sh_comb = np.array(arr.shape[1:])  # Shape of final combined image

    # == organize offsets =============================================== #
    if isinstance(offsets, str):
        raise ValueError(
            "If given arr, offsets must be a list with len=arr.shape[0]."
        )
    elif offsets is None:
        offsets = np.zeros(sh_comb)
    else:
        offsets = np.atleast_1d(offsets)  # N x ndim array
        if offsets.shape[0] != arr.shape[0]:
            raise ValueError(
                "If given arr, offset.shape[0] == arr.shape[0]"
            )
    # ------------------------------------------------------------------- #
    offmins = np.min(offsets, axis=0)
    offmaxs = np.max(offsets, axis=0)
    sh_comb = sh_comb + offmaxs - offmins
    mem2use = np.prod(sh_comb) * np.dtype(dtype).itemsize

    if (memused + mem2use) > memlimit:
        raise ValueError(
            "If given arr, I cannot handle memory issue. Use fpaths "
            + "or increase the memlimit."
        )

    # N x ndim of image sizes for each of N images.
    sizes = np.tile(arr.shape[1:], arr.shape[0]).reshape(arr.shape[0], -1)
'''
