# vla_polcal
Polarization calibration pipeline for VLA continuum visibilities

This version requires the calibration outputs from the NRAO VLA pipeline to be present in the working directory, in order to un-do the parallactic angle correction done by the pipeline. The pipeline also takes into account the recent flaring of 3C147, and fits a polynomial to the flux density and polarization angle of known polarization calibrators.

Operations:
- (optional) Reverts the parallactic angle correction applied by the NRAO vla pipeline
- (optional) Performs gain calibration on the leakage calibrator (useful for resolved sources)
- Performs statistical outlier flagging on cross-hands
- Least-squares fitting of total flux, polarization fraction, and polarization angle using nth order polynomials for polarization angle calibrator
- Sets the frequency-dependent flux density scale of leakage and angle calibrators
- Calibrates the (per-baseband) cross-hand delays
- Calibrates the (per-baseband) leakage
- Splits the MS and applies new statistical weights

# Usage:<br />
-Create a config, using the example config provided<br />

-```python polcal_steps.py <msin> <msout> <msout_target> <output_directory> config.txt```<br />

This will output a fully polarimetric calibrated measurement set containing your science target. It is expected to run self-calibration after this.

# Software requirements:<br />

CASA<br />
Python3<br />
singularity<br />

# Example outputs
Example polynomic functions fitted to the total flux, polarization angle, and linearly polarized fraction against frequency, for the calibrator 3C 138 (used to calibrate the polarization angle). <br />
<img width="640" height="480" alt="FluxvFreq" src="https://github.com/user-attachments/assets/8a0fcb56-8d30-438c-be65-af736be750ef" />
<img width="640" height="480" alt="LinPolAnglevFreq" src="https://github.com/user-attachments/assets/0278c0b7-76e9-4eb0-bf3a-bbfa62545750" />
<img width="640" height="480" alt="LinPolFracvFreq" src="https://github.com/user-attachments/assets/194b9e24-a95c-4e9b-a73f-c44f70570cfa" />
