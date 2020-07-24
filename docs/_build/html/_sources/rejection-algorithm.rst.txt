.. currentmodule:: imcombinepy

.. _rejection-algorithm:

########################
The Rejection Algorithms
########################
.. _IMCOMBINE: https://iraf.net/irafhelp.php?val=imcombine&help=Help+Page

This is a documentation for explaining the rejection algorithms. The ``reject`` module is **not intended to be used by the users**.

**********************
Implemented Algorithms
**********************

Among the 6 rejection algorithms in IRAF `IMCOMBINE`_, ``sigclip``, ``ccdclip``, ``minmax``, and ``pclip`` are implemented (``minmax`` and ``pclip`` will come soon).

::

    none      - No rejection
    sigclip   - Reject pixels using a sigma clipping algorithm
    minmax    - Reject the nlow and nhigh pixels
    pclip     - Reject pixels using sigma based on percentiles
    ccdclip   - Reject pixels using CCD noise parameters
    crreject  - Reject only positive pixels using CCD noise parameters
    avsigclip - Reject pixels using an averaged sigma clipping algorithm


``avsigclip`` and ``crreject`` are not implemented separately yet, because

    #. ``avsigclip`` is ``ccdclip`` with readout noise is fixed 0 (``rdnoise = 0``).
    #. ``crreject`` is ``ccdclip`` with infinite ``lsigma`` (``sigma_lower``).


Parameter Mappings with IRAF
============================

Below is a summary of parameters that are different between IRAF and ``imcombinepy``.

+------------------------+------------------------------------------------------------------------------+--------------------------+
| IRAF `IMCOMBINE`_      | ``imcombinepy``                                                              | affected algorithms      |
+------------------------+------------------------------------------------------------------------------+--------------------------+
| ``nkeep``              | ``nkeep``, ``maxrej``                                                        | ALL                      |
+------------------------+------------------------------------------------------------------------------+--------------------------+
| ``mclip``              | absorbed into ``cenfunc`` (e.g., ``cenfunc="median"`` is ``mclip+`` of IRAF) | ``sigclip``, ``ccdclip`` |
+------------------------+------------------------------------------------------------------------------+--------------------------+
| ``lsigma``, ``hsigma`` | ``sigma`` (``sigma_lower`` and ``sigma_upper``)                              | ``sigclip``, ``ccdclip`` |
+------------------------+------------------------------------------------------------------------------+--------------------------+
| ``sigscale``, ``grow`` | **not implemented**                                                          | ALL                      |
+------------------------+------------------------------------------------------------------------------+--------------------------+

In IRAF, only one of ``nkeep`` or ``maxrej`` can be set (negative ``nkeep`` in IRAF is ``maxrej`` in ``imcombinepy``). By default, ``imcombinepy`` has ``nkeep=3`` and ``maxrej`` 99.999...%, i.e., after the rejection, at least 3 pixels must be remaining. Otherwise, it will revert to the previous iteration. The IRAF default is identical to this when ``reject=sigclip``, alghouth the default of ``nkeep`` is stated as ``1`` (because it automatically sets ``nkeep>=3`` if ``reject=sigclip``). As thr standard deviation in numpy or bottleneck both can cope with array smaller than 3 values, I did not implement to change ``nkeep``.

.. note::
    If you have 7 images, and at one position, you have only 4 available pixels to combine. If you set ``nkeep=3``, the rejection will happen, and few out of 4 pixels *may* be rejected. If you additionally set ``maxrej=3``, for instance, the rejection will not happen because if rejection happens, more than 3 pixels will have been rejected. It is not supported in IRAF to specify both of these.


**************
Documentations
**************


CCD noise clipping
==================
When ``reject`` is one of ``['ccd', 'ccdclip', 'ccdc']``.

The central value (``cen``) is first determined by ``cenfunc``. The "sigma" value is calculated by the CCD noise equation:

.. math::
    \mathtt{err}^2 = I_e + \mathtt{rdnoise}^2 + \mathtt{snoise} \times I_e
    = (1 + \mathtt{snoise}) \times I_e + \mathtt{rdnoise}^2

Here, :math:`I_e` is the central value determined by ``cenfunc`` at each iteration in rejection, in **electron, not DN**. Because it is easier to play with gain-corrected values (in usual CCD, that means in the unit of electron), since it is electron numbers that is Poisson distributed, I made the internal API to use gain-corrected arrays. ``rdnoise`` is assumed to be in electron and ``gain`` in electron per DN (ADU), so-called ``EPADU`` in IRAF.

When the frames have different ``gain``, it does not matter for finding the ``cen`` and ``err``. However, when ``snoise`` and/or ``rdnoise`` differ, the mean of each is used for estimating the ``err`` according to the equation above. After the first iteration, any value ``arr < sigma_lower*std`` or ``sigma_upper*std < arr`` is replaced with ``nan`` (use ``copy=True`` if you are using :func:`~imcombinepy.combine.ndcombine`). At each position (pixel for 2-D image, voxel for 3-D image, etc), the number of rejected points are calculated. If this exceeds ``maxrej`` or if the remaining non-NaN value is fewer than ``nkeep``, the rejection is reverted to the previous iteration.

Sometimes ``maxiters = 0`` is given, and in such case, the lower and upper bounds are nothing but ``nanmin`` and ``nanmax`` at the position, and number of iteration is 0. It sometimes happens that the number of remaining pixels at the position is fewer than ``nkeep`` even before any rejection due to the severe masking or many NaN values at the position. Similarly ``maxrej`` condition may be met even before any rejection. Then only ``nanmin`` and ``nanmax`` will be given as lower and upper bounds as above. The ``o_code`` will hint what happened.

IRAF documentation states that (1) the iteration is repeated until no further rejection occurs and (2) the minimum and maximum pixels are excluded at the very first iteration. In ``imcombinepy`` implementation, however, ``maxiters`` is a free parameter and the minimum/maximum pixels are **not** rejected in the initial stage.

I couldn't understand the description on ``sigscale`` in IRAF, so what I implemented here is to use :math:`I_e = \mathtt{(cen + zero\_ref)*scale\_ref}` for representative (mean of) zeros and scales to revert the zeroing and scaling (for all iterations).



Sigma-clipping
==============
When ``reject`` is one of ``['sig', 'sc', 'sigclip', 'sigma', 'sigma clip', 'sigmaclip']``.

The central value is first determined by ``cenfunc``. The "sigma" value is calculated by ``nanstd`` with the given ``ddof``. After the first iteration, any value ``arr < sigma_lower*std`` or ``sigma_upper*std < arr`` is replaced with ``nan`` (use ``copy=True`` if you are using :func:`~imcombinepy.combine.ndcombine`).. At each position (pixel for 2-D image, voxel for 3-D image, etc), the number of rejected points are calculated. If this exceeds ``maxrej`` or if the remaining non-NaN value is fewer than ``nkeep``, the rejection is reverted to the previous iteration.

Sometimes ``maxiters = 0`` is given, and in such case, the lower and upper bounds are nothing but ``nanmin`` and ``nanmax`` at the position, and number of iteration is 0. It sometimes happens that the number of remaining pixels at the position is fewer than ``nkeep`` even before any rejection due to the severe masking or many NaN values at the position. Similarly ``maxrej`` condition may be met even before any rejection. Then only ``nanmin`` and ``nanmax`` will be given as lower and upper bounds as above. The ``o_code`` will hint what happened.

In IRAF, it is stated that for positions where fewer than ``nkeep`` pixels after the rejection, the restoration process is done rather than simple "rewinding to the previous iteration" scheme (used in ``imcombinepy``). If ``irafmode=True`` is used, ``imcombinepy`` tries to reproduce IRAF results, but somehow the results do not match perfectly.

.. note::
    This is a direct excerpt from IRAF `IMCOMBINE`_:

    * After rejection the number of retained pixels is checked against the ``nkeep`` parameter. If there are fewer pixels retained than specified by this parameter the pixels with the smallest residuals in absolute value are added back. If there is more than one pixel with the same absolute residual (for example the two pixels about an average or median of two will have the same residuals) they are all added back even if this means more than nkeep pixels are retained. Note that the nkeep parameter only applies to the pixels used by the clipping rejection algorithm and does not apply to threshold or bad pixel mask rejection.



Percentile clipping
===================
When ``reject`` is one of ``['pclip', 'pc', 'percentile']``.

Not available yet.

Minmax clipping
===============
When ``reject`` is one of ``['mm', 'minmax']``.

Not available yet.


*************
Reference/API
*************

.. automodule:: imcombinepy.reject

.. toctree::
    :maxdepth: 3

    api/reject.rst

