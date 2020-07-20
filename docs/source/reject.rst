.. module:: imcombinepy

.. _reject:

########################
The Rejection Algorithms
########################

**********************
Implemented Algorithms
**********************

Among the 6 rejection algorithms in IRAF `IMCOMBINE
<https://iraf.net/irafhelp.php?val=imcombine&help=Help+Page>`_,

::

    none      - No rejection
    sigclip   - Reject pixels using a sigma clipping algorithm
    minmax    - Reject the nlow and nhigh pixels
    pclip     - Reject pixels using sigma based on percentiles
    ccdclip   - Reject pixels using CCD noise parameters
    crreject  - Reject only positive pixels using CCD noise parameters
    avsigclip - Reject pixels using an averaged sigma clipping algorithm


``avsigclip`` and ``crreject`` are not implemented as

    #. ``avsigclip`` is ``ccdclip`` with readout noise is fixed 0 (``rdnoise = 0``).
    #. ``crreject`` is ``ccdclip`` with infinite ``lsigma`` (``sigma_lower``).

******************
Parameter mappings
******************
Below is a summary of parameters that are different between IRAF and ``imcombinepy``.

+------------------------+------------------------------------------------------------------------------+--------------------------+
| IRAF                   | ``imcombinepy``                                                              | affected algorithms      |
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
    If you have 7 images, and at one position, you have only 4 available pixels to combine. If you set ``nkeep=3``, the rejection will happen, and few out of 4 pixels *may* be rejected. If you additionally set ``maxrej=3``, for instance, the rejection will not happen because if rejection happens, more than 3 pixels will have been rejected.

**************
Documentations
**************

Sigma-clipping
--------------
When ``reject`` is one of ``['sig', 'sc', 'sigclip', 'sigma', 'sigma clip', 'sigmaclip']``.

.. autofunction:: imcombinepy.reject.sigclip_mask


CCD noise clipping
------------------
When ``reject`` is one of ``['ccd', 'ccdclip', 'ccdc']``.

Not available yet.

Percentile clipping
-------------------
When ``reject`` is one of ``['pclip', 'pc', 'percentile']``.

Not available yet.

Minmax clipping
---------------
When ``reject`` is one of ``['mm', 'minmax']``.

Not available yet.
