import glob
from pathlib import Path

import bottleneck as bn
import numpy as np
from astropy.io import fits

from .reject import sigclip_mask
from .util import (_get_combine_shape, _set_calc_zsw, _set_cenfunc,
                   _set_combfunc, _set_int_dtype, _set_keeprej, _set_mask,
                   _set_reject, _set_sigma, update_hdr, write2fits)

__all__ = ["imcombine", "ndcombine"]

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


def imcombine(
        fpaths=None, fpattern=None, mask=None, ext=0,
        fits_section=None,
        blank=np.nan,
        offsets=None,
        thresholds=[-np.inf, np.inf],
        zero=None, scale=None, weight=None, statsec=None,
        zero_kw={'cenfunc': 'median', 'stdfunc': 'std', 'std_ddof': 1},
        scale_kw={'cenfunc': 'median', 'stdfunc': 'std', 'std_ddof': 1},
        scale_sample=None, zero_sample=None,
        reject=None,
        cenfunc='median',
        sigma=[3., 3.], maxiters=3, ddof=1, nkeep=1, maxrej=None,
        n_minmax=[1, 1],
        rdnoise=0., gain=1., snoise=0.,
        pclip=-0.5,
        logfile=None,
        combine='average',
        dtype='float32',
        memlimit=2.5e+9,
        verbose=False,
        full=False,
        imcmb_key='$I', exposure_key="EXPTIME",
        output=None, output_mask=None, output_nrej=None,
        output_sigma=None, output_low=None, output_upp=None,
        output_rejcode=None,
        **kwargs
):
    '''A helper function for ndcombine to cope with FITS files.

    Parameters
    ----------
    fpaths : list-like of path-like, optional.
        The list of file paths to be combined. These must be FITS files.
        One and only one of ``fpaths`` or ``fpattern`` must be provided.

    fpattern : str, optional.
        The ``os.glob`` pattern for files (e.g., ``"2020*[012].fits"``).
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
        images and ``m`` is the dimension of the images (offsets in x,
        y, z, ... order, not pythonic order), and it is directly
        regarded as the "raw offsets". If ``str``, the "raw offsets" are
        obtained by the followings:
            * ``"wcs"`` or ``"world"``:
                Raw offsets are the ``CRPIX`` values in the header.
            * ``"physical"``, ``"phys"``, or ``"phy"``:
                Raw offsets are the ``LTV`` values in the header. The
                physical coordinate system is defined by the IRAF-like
                ``LTM``/``LTV`` keywords define the offsets. Currently,
                only the cases when ``LTMi_j`` is 0 or 1 can be managed.
                Otherwise, we need scaling and it is not supported now.
        For these cases, the raw offsets for each frame is nothing but
        an ``m``-D tuple consists of ``offset_raw[i] = CRPIX{m-i}`` or
        ``LTV{m-i}[_{m-i}]``. The reason to subtract ``i`` is because
        python has ``z, y, x`` order of indexing while WCS information
        is in ``x, y, z`` order.

        The raw offsets are then modified such that the minimum offsets
        in each axis becomes zero (in pythonic way,
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
        The "zero" value to subtract from each image _after_
        thresholding, but _before_ scaling/rejection/combination. If an
        array, it is directly subtracted from each image, (so it must
        have size identical to the number of images). If ``str``:
            * ``'avg'``, ``'average'``, ``'mean'``:
                Subtract the average value of each image from itself.
            * ``'med'``, ``'medi'``, ``'median'``:
                Subtract the median value of each image from itself.
            * ``'avg_sc'``, ``'average_sc'``, ``'mean_sc'``:
                Subtract the sigma-clipped average value of each image
                from itself.
            * ``'med_sc'``, ``'medi_sc'``, ``'median_sc'``:
                Subtract the sigma-clipped median value of each image
                from itself.
        For options for sigma-clipped statistics, see ``zero_kw``.

        .. note::
            By using ``zero="med_sc"``, the user can crudely subtract
            sky value from each frame.

    scale : str or 1-d array
        The way to scale each image _after_ thresholding/zeroing, but
        _before_ rejection/combination. If an array, it is directly
        understood as the "raw scales", and it must have size identical
        to the number of images. If ``str``:
            * ``"exp"``, ``"expos"``, ``"exposure"``, or ``"exptime"``:
                Uses the exposure time for raw scale. The header keyword
                specified by ``exposure_key`` will be searched for each
                input FITS file.
            * ``'avg'``, ``'average'``, ``'mean'``:
                Uses the average value of each image as raw scales.
            * ``'med'``, ``'medi'``, ``'median'``:
                Uses the median value of each image as raw scales.
            * ``'avg_sc'``, ``'average_sc'``, ``'mean_sc'``:
                Uses the sigma-clipped average value of each image as
                raw scales.
            * ``'med_sc'``, ``'medi_sc'``, ``'median_sc'``:
                Uses the sigma-clipped median value of each image as raw
                scales.
        The true scale is obtained by ``scales / scales[0]``, following
        IRAF's convention. For options for sigma-clipped statistics,
        see ``scale_kw``.

        .. note::
            Using ``scale="avg_sc"`` is useful for flat combining.

    zero_kw, scale_kw : dict
        Used only if ``scale`` or ``zero`` are sigma-clipped mean,
        median, etc (ending with ``_sc`` such as ``median_sc``,
        ``avg_sc``). The keyword arguments for astropy's
        `~astropy.stats.sigma_clipped_stats`. By default,
        ``std_ddof=1``, which is different from that of original
        ``sigma_clipped_stats``.

        .. warning::
            Do not specify ``axis``.

    sigma : 2-float list-like, optional.
        The sigma-factors to be used for sigma-clip rejeciton in
        ``(sigma_lower, sigma_upper)``. Defaults to ``(3, 3)``, which
        means 3-sigma clipping from the "sigma" values determined by the
        method specified by ``reject``.

    maxiters : int, optional.
        The maximum number of iterations to do the rejection (for
        sigma-clipping). It is silently converted to ``int`` if it is
        not.

    ddof : int, optional.
        The delta-degrees of freedom (see `~numpy.std`). It is silently
        converted to ``int`` if it is not.

    nkeep : float or int, optional.
        The minimum number of pixels that should be left after
        rejection. If ``nkeep < 1``, it is regarded as fraction of the
        total number of pixels along the axis to combine. This
        corresponds to _positive_ ``nkeep`` parameter of IRAF IMCOMBINE.
        If number of remaining non-nan value is fewer than ``nkeep``,
        the masks at that position will be reverted to the previous
        iteration, and rejection code will be added by number 4.

    maxrej : float or int, optional.
        The maximum number of pixels that can be rejected during the
        rejection. If ``maxrej < 1``, it is regarded as fraction of the
        total number of pixels along the axis to combine. This
        corresponds to _negative_ ``nkeep`` parameter of IRAF IMCOMBINE.
        In IRAF, only one of ``nkeep`` and ``maxrej`` can be set.
        If number of rejected pixels at a position exceeds ``maxrej``,
        the masks at that position will be reverted to the previous
        iteration, and rejection code will be added by number 8.

    cenfunc : str, optional.
        The centering function to be used in rejection algorithm.

          * median if  ``cenfunc in ['med', 'medi', 'median']``
          * average if ``cenfunc in ['avg', 'average', 'mean']``
          * lower median if ``cenfunc in ['lmed', 'lmd', 'lmedian']``

        The lower median means the median which takes the lower value
        when even number of data is left. This is suggested to be robust
        against cosmic-ray hit according to IRAF IMCOMBINE manual.

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
    comb : `~astropy.fits.PrimaryHDU`
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
    if offsets in ['world', 'wcs']:
        # w_ref = WCS(hdr0)
        # cen_ref = np.array([hdr0[f'NAXIS{i+1}']/2 for i in range(ndim)])
        use_wcs = True
        offsets = np.zeros((ncombine, ndim))
    elif offsets in ['physical', 'phys', 'phy']:
        use_phy = True
        offsets = np.zeros((ncombine, ndim))
    elif offsets is None:
        use_wcs = False
        use_phy = False
        offsets = np.zeros((ncombine, ndim))
    else:
        if offsets.shape[0] != ncombine:
            raise ValueError("offset.shape[0] must be num(images)")

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
    mask_full = _set_mask(arr_full, mask)
    for i, (_fpath, _offset, _size) in enumerate(zip(fpaths,
                                                     offsets,
                                                     sizes)):
        # -- Set slice ------------------------------------------------------ #
        slices = [i]
        # offset & size at each dimension
        for offset_i, size_i in zip(_offset, _size):
            slices.append(slice(offset_i, offset_i + size_i, None))
        # ------------------------------------------------------------------- #

        # -- Set mask ------------------------------------------------------- #
        hdul = fits.open(_fpath)
        try:  # load MASK from FITS file if exists
            _mask = hdul["MASK"].data.astype('bool')
        except KeyError:
            _mask = np.zeros(hdul[ext].data.shape, dtype=bool)

        if mask is not None:
            _mask |= mask[i, ]
        # ------------------------------------------------------------------- #

        arr_full[slices] = hdul[ext].data
        mask_full[slices] = _mask

    if verbose:
        print("All FITS loaded, rejection & combination starts", end='... ')

    # == Combine with rejection! ============================================ #
    comb, sigma, mask_rej, mask_thresh, low, upp, nit, rejcode = ndcombine(
        arr=arr_full,
        mask=mask_full,
        logfile=logfile,
        combine=combine,
        reject=reject,
        scale=scale,
        zero=zero,
        scale_kw=scale_kw,
        zero_kw=zero_kw,
        weight=weight,
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

    update_hdr(hdr0, ncombine, imcmb_key=imcmb_key, imcmb_val=imcmb_val)
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

    if full:
        return (comb, sigma, mask_total, mask_rej, mask_thresh,
                low, upp, nit, rejcode)
    else:
        return comb


# --------------------------------------------------------------------------- #
def ndcombine(
        arr, mask=None, copy=True,
        offsets=None,
        thresholds=[-np.inf, np.inf],
        zero=None, scale=None, weight=None, statsec=None,
        zero_kw={'cenfunc': 'median', 'stdfunc': 'std', 'std_ddof': 1},
        scale_kw={'cenfunc': 'median', 'stdfunc': 'std', 'std_ddof': 1},
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
        verbose=False,
        full=False
):
    ''' Combines the given arr assuming no additional offsets.

    Parameters
    ----------
    arr : ndarray
        The array to be combined along axis 0.

    mask : ndarray, optional.
        The mask of bad pixels. If given, it must satisfy
        ``mask.shape[0]`` identical to the number of images.

    copy : bool, optional.
        Whether to copy the input array. Set to ``True`` if you want to
        keep the original array unchanged.

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
        The "zero" value to subtract from each image _after_
        thresholding, but _before_ scaling/rejection/combination. If an
        array, it is directly subtracted from each image, (so it must
        have size identical to the number of images). If ``str``:
            * ``'avg'``, ``'average'``, ``'mean'``:
                Subtract the average value of each image from itself.
            * ``'med'``, ``'medi'``, ``'median'``:
                Subtract the median value of each image from itself.
            * ``'avg_sc'``, ``'average_sc'``, ``'mean_sc'``:
                Subtract the sigma-clipped average value of each image
                from itself.
            * ``'med_sc'``, ``'medi_sc'``, ``'median_sc'``:
                Subtract the sigma-clipped median value of each image
                from itself.
        For options for sigma-clipped statistics, see ``zero_kw``.

        .. note::
            By using ``zero="med_sc"``, the user can crudely subtract
            sky value from each frame.

    scale : str or 1-d array
        The way to scale each image _after_ thresholding/zeroing, but
        _before_ rejection/combination. If an array, it is directly
        understood as the "raw scales", and it must have size identical
        to the number of images. If ``str``:
            * ``"exp"``, ``"expos"``, ``"exposure"``, or ``"exptime"``:
                Uses the exposure time for raw scale. The header keyword
                specified by ``exposure_key`` will be searched for each
                input FITS file.
            * ``'avg'``, ``'average'``, ``'mean'``:
                Uses the average value of each image as raw scales.
            * ``'med'``, ``'medi'``, ``'median'``:
                Uses the median value of each image as raw scales.
            * ``'avg_sc'``, ``'average_sc'``, ``'mean_sc'``:
                Uses the sigma-clipped average value of each image as
                raw scales.
            * ``'med_sc'``, ``'medi_sc'``, ``'median_sc'``:
                Uses the sigma-clipped median value of each image as raw
                scales.
        The true scale is obtained by ``scales / scales[0]``, following
        IRAF's convention. For options for sigma-clipped statistics,
        see ``scale_kw``.

        .. note::
            Using ``scale="avg_sc"`` is useful for flat combining.

    zero_kw, scale_kw : dict
        Used only if ``scale`` or ``zero`` are sigma-clipped mean,
        median, etc (ending with ``_sc`` such as ``median_sc``,
        ``avg_sc``). The keyword arguments for astropy's
        `~astropy.stats.sigma_clipped_stats`. By default,
        ``std_ddof=1``, which is different from that of original
        ``sigma_clipped_stats``.

        .. warning::
            Do not specify ``axis``.

    sigma : 2-float list-like, optional.
        The sigma-factors to be used for sigma-clip rejeciton in
        ``(sigma_lower, sigma_upper)``. Defaults to ``(3, 3)``, which
        means 3-sigma clipping from the "sigma" values determined by the
        method specified by ``reject``.

    maxiters : int, optional.
        The maximum number of iterations to do the rejection (for
        sigma-clipping). It is silently converted to ``int`` if it is
        not.

    ddof : int, optional.
        The delta-degrees of freedom (see `~numpy.std`). It is silently
        converted to ``int`` if it is not.

    nkeep : float or int, optional.
        The minimum number of pixels that should be left after
        rejection. If ``nkeep < 1``, it is regarded as fraction of the
        total number of pixels along the axis to combine. This
        corresponds to _positive_ ``nkeep`` parameter of IRAF IMCOMBINE.
        If number of remaining non-nan value is fewer than ``nkeep``,
        the masks at that position will be reverted to the previous
        iteration, and rejection code will be added by number 4.

    maxrej : float or int, optional.
        The maximum number of pixels that can be rejected during the
        rejection. If ``maxrej < 1``, it is regarded as fraction of the
        total number of pixels along the axis to combine. This
        corresponds to _negative_ ``nkeep`` parameter of IRAF IMCOMBINE.
        In IRAF, only one of ``nkeep`` and ``maxrej`` can be set.
        If number of rejected pixels at a position exceeds ``maxrej``,
        the masks at that position will be reverted to the previous
        iteration, and rejection code will be added by number 8.

    cenfunc : str, optional.
        The centering function to be used in rejection algorithm.

          * median if  ``cenfunc in ['med', 'medi', 'median']``
          * average if ``cenfunc in ['avg', 'average', 'mean']``
          * lower median if ``cenfunc in ['lmed', 'lmd', 'lmedian']``

        The lower median means the median which takes the lower value
        when even number of data is left. This is suggested to be robust
        against cosmic-ray hit according to IRAF IMCOMBINE manual.

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

    '''
    if copy:
        arr = arr.copy()

    if np.array(arr).ndim == 1:
        raise ValueError("1-D array combination is not supported!")

    _mask = _set_mask(arr, mask)  # _mask = propagated through this function.
    sigma_lower, sigma_upper = _set_sigma(sigma)
    nkeep, maxrej = _set_keeprej(arr, nkeep, maxrej, axis=0)
    cenfunc = _set_cenfunc(cenfunc)
    reject = _set_reject(reject)
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
    if (thresholds[0] != -np.inf) and (thresholds[1] != np.inf):
        mask_thresh = (arr < thresholds[0]) | (arr > thresholds[1])
        _mask |= mask_thresh
    elif (thresholds[0] == -np.inf):
        mask_thresh = (arr > thresholds[1])
        _mask |= mask_thresh
    elif (thresholds[1] == np.inf):
        mask_thresh = (arr < thresholds[0])
        _mask |= mask_thresh
    else :
        mask_thresh = np.zeros(arr.shape).astype(bool)
        # no need to update _mask

    # Backup the pixels which are rejected by thresholding for future
    # restoration (see below) for debugging purpose.
    backup_thresh = arr[mask_thresh]
    arr[_mask] = np.nan
    # ----------------------------------------------------------------------- #

    # == 02 - Calculate scale, zero, weights ================================ #
    # This should be done before rejection but after threshold masking..
    # If it were done before threshold masking, it must have been much easier.

    if scale is None:
        scale = np.ones(arr.shape[0])
    if zero is None:
        zero = np.zeros(arr.shape[0])
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
        scales /= scales[0]  # So that normalize 1.000 for the 0-th image.
    if calc_w:  # TODO: Needs update to match IRAF's...
        if fun_w == fun_s:
            weights = scales.copy()
        elif fun_w == fun_z:
            weights = zeros.copy()
        else:
            for i in range(arr.shape[0]):
                weight.append(fun_w(arr[i, ]))
            weights = np.array(weights)

    for i in range(arr.shape[0]):
        arr[i, ] = (arr[i, ] - zeros[i])/scales[i]
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
