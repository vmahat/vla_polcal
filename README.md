# vla_polcal
Polarization calibration pipeline for VLA continuum visibilities

This version requires the calibration outputs from the NRAO VLA pipeline to be present in the working directory, in order to un-do the parallactic angle correction done by the pipeline. The pipeline also takes into account the recent flaring of 3C147, and fits a polynomial to the flux density and polarization angle of known polarization calibrators.

Usage:<br />
-Create a config, using the example config provided<br />

-```python polcal_steps.py <msin> <msout> <msout_target> <output_directory> config.txt```<br />

Example run on JVLA B-array Ku-band data observing a number of 3C sources and includes a polarization leakage calibrator and polarization angle calibrator<br />
```python polcal_steps.py /beegfs/general/mahatmav/lofar/long_baseline/3C123/vla/Kuband/B-array/21B-036.sb40054536.eb40480205.59488.42185204861/21B-036.sb40054536.eb40480205.59488.42185204861.ms /beegfs/general/mahatmav/lofar/long_baseline/3C123/vla/Kuband/B-array/21B-036.sb40054536.eb40480205.59488.42185204861/polcal/21B-036.sb40054536.eb40480205.59488.42185204861_polcal.ms /beegfs/general/mahatmav/lofar/long_baseline/3C123/vla/Kuband/B-array/21B-036.sb40054536.eb40480205.59488.42185204861/polcal/21B-356_Ku_calib_3C123.ms /beegfs/general/mahatmav/lofar/long_baseline/3C123/vla/Kuband/B-array/21B-036.sb40054536.eb40480205.59488.42185204861/polcal/ config_3C123_Ku_B.txt```<br />

This will output a fully polarimetric calibrated measurement set containing your science target. It is expected to run self-calibration after this.

Software requirements:<br />

CASA<br />
Python3<br />
singularity<br />
