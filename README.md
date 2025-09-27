# vla_polcal
Polarization calibration pipeline for VLA continuum visibilities

This version requires the calibration outputs from the NRAO VLA pipeline to be present in the working directory, in order to un-do the parallactic angle correction done by the pipeline. The pipeline also takes into account the recent flaring of 3C147, and fits a polynomial to the flux density and polarization angle of known polarization calibrators.

Usage:
-Create a config, using the example config provided
-run `python polcal_steps.py <msin> <msout> <msout_target> <output_directory> config.txt`

This will output a fully polarimetric calibrated measurement set containing your science target. It is expected to run self-calibration after this.

Software requirements:

CASA
Python3
singularity
