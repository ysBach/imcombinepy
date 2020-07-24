.. imcombinepy documentation master file, created by
   sphinx-quickstart on Sun Jun 28 15:14:46 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _imcombinepy:

#############################
``imcombinepy`` Documentation
#############################

.. _IMCOMBINE: https://iraf.net/irafhelp.php?val=imcombine&help=Help+Page

A python package to replace IRAF `IMCOMBINE`_
with both python and CLI interface using bottleneck.


************************
The Image Combining Flow
************************

:func:`~imcombinepy.combine.ndcombine` function is intended to be used to help :func:`~imcombinepy.combine.fitscombine`. It can be used to combine an ndarray along axis 0. One may use it such as ``arr = [fits.open(fpath)[0].data for fpath in fpaths]`` so that the combination along axis 0 is what the user may want. The function does the following tasks in this order:

#. Mask pixels outside ``thresholds``.
#. Scales the frames by ``scale``, ``zero`` and related arguments.
#. Reject pixels based on ``reject`` and related arguments (see algorithm documentation).
#. Combine images based on ``combine``.


The main function, :func:`~imcombinepy.combine.fitscombine` does the following tasks in this order:

#. Determine the strategy for proper memory limit (not implemented yet).
#. Extract information from header (exposure time, gain, readout noise, sensitivity noise, and WCS if they're needed).
#. Determine the offset of each frame (if ``offsets`` is given).
#. Make a new array which has offset-ed array at each slice along axis 0. Blank pixels are filled as ``np.nan``.
#. Prepare zero, and scale of each frame (if they are given). If FITS files have ``'MASK'`` extension, load it and propagate it with the ``mask`` input by the user.
#. Pass these arguments to `~imcombinepy.combine.ndcombine`.
#. Convert the combined image to FITS format (`astropy.io.fits.PrimaryHDU`) and update header to incorporate with proper WCS information.
#. Save the auxiliary files if output paths are specified.


*************
Documentation
*************
.. toctree::
   :maxdepth: 2

   rejection-algorithm
   IRAFcomparison


****
APIs
****
.. automodule:: imcombinepy.combine

.. toctree::
   :maxdepth: 2

   api/combine.rst

.. automodule:: imcombinepy.reject

.. toctree::
   :maxdepth: 2

   api/reject.rst



*******
LICENSE
*******
BSD 3-Clause License

.. toctree::
   :maxdepth: 2

   license
