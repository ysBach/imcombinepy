.. currentmodule:: imcombinepy.reject

.. _rejection:

Rejection APIs
**************
.. _bench_isnan.md: https://github.com/ysBach/imcombinepy/tree/master/bench/bench_isnan.md

See `bench_isnan.md`_ why ``nanXXX`` functions are used (e.g., not ``median`` instead of ``nanmedian``).



.. note::
    Tips to use the returned masks from the functions below:

    #. The number of rejected points: ``np.count_nonzero(o_mask, axis=0)``.

    #. The original ``mask`` is propagated, so the mask for pixels masked *purely* from rejection algorithm is obtained by ``o_mask^mask`` (which is the ``mask_rej`` of :func:`~imcombinepy.combine.fitscombine` or :func:`~imcombinepy.combine.ndcombine`), because the input ``mask`` is a subset of ``o_mask``.

:func:`ccdclip_mask`
====================
.. autofunction:: imcombinepy.reject.ccdclip_mask

:func:`sigclip_mask`
====================
.. autofunction:: imcombinepy.reject.sigclip_mask
