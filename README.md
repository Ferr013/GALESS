# GaLeSS (Galaxy Lenses Search Statistics)

GaLeSS forecasts the number of identifiable galaxy–galaxy strong lenses in a
photometric survey. It predicts their lens and source redshifts, lens velocity
dispersions, Einstein radii and source magnitudes. The survey depth, seeing,
signal-to-noise threshold and treatment of foreground light enter the selection.
The model and its comparison with observed lens searches are described in
[Ferrami & Wyithe (2024)](https://arxiv.org/abs/2404.03143).

## Installation

From this checkout, run `python -m pip install .` in a Python 3.9–3.11 environment.
The core package uses the dependencies pinned in [requirements.txt](requirements.txt).

The halo occupation model includes halo mass and power tables at redshift zero
and from 1.2 to 2.2 in steps of 0.1. To generate tables at other redshifts,
install `python -m pip install '.[hod]'`. Generated tables go in
`~/.cache/galess/HMF_tables` or the directory set by `GALESS_HMF_CACHE`.

## Calculate a survey

This example runs the Euclid Wide VIS forecast with a singular isothermal
ellipsoid (SIE) lens and complete subtraction of lens light.

```python
import numpy as np
from galess.LensStat import lens_stat as ls
from galess.Utils import ls_utils as utils

p = utils.read_survey_params('EUCLID Wide VIS')
sigma = np.linspace(100, 400, 31)  # km/s
zl = np.arange(0, 2.5, 0.1)
zs = np.arange(0, 5.4, 0.2)
M = np.linspace(-25, -13, 25)      # absolute source UV magnitude

counts, theta_E, by_magnitude = ls.calculate_num_lenses_and_prob(
    sigma, zl, zs, M, p['limit'], p['area'], p['seeing'], 20,
    p['exp_time_sec'], p['sky_bckgnd_m'], p['zero_point_m'], p['photo_band'],
    mag_cut=p['cut'], arc_mu_threshold=3, LENS_LIGHT_FLAG=False, SIE_FLAG=True)
print(f'{counts.sum():,.0f} identifiable lenses')  # 149,922
```

`counts` and `theta_E` have axes `(zs, sigma, zl)`. Radii are in arcseconds.
`by_magnitude` adds a fourth axis ordered from bright to faint. The count
matrix integrates that axis up to the survey magnitude cut. The serial
forecast keeps the supplied redshift grid and leaves excluded bins at zero.
Use equally spaced axes when interpreting the normalized one-dimensional
densities returned by `get_N_and_P_projections`.

Set `LENS_LIGHT_FLAG=True` to model detection through the foreground light
profile, or `SIE_FLAG=False` for a singular isothermal sphere (SIS). Pass
`Phi_vel_disp` or `LF_func` to specify another lens or source population.
Survey definitions are in [surveys_params](galess/data/surveys_params).
The original SuGOHI preset covers 178 square degrees. Select
`SUBARU HSC SuGOHI i band 1400deg2` for the wider footprint and deeper cut.

`galess.LensStat.parallel_ls.calculate_num_lenses_and_prob` returns the
count matrix for observable source planes only. `ncores` sets the number
of workers. When using multiprocessing in a script, put the call inside
`if __name__ == '__main__':`.

## Examples

| Notebook | Calculation |
| --- | --- |
| [Basic survey forecast](examples/basic_example.ipynb) | Compare lens-light selections, change the VDF and magnitude cut, and plot the distributions. |
| [Euclid lens population](examples/Euclid_lenses.ipynb) | Draw a reproducible mock catalog, inspect joint distributions and compare marginals with Collett (2015). |
| [Rare lenses and halo occupations](examples/other_figures.ipynb) | Count detectable cusps and quads, forecast double-source systems in two bands, and calculate host masses and satellite fractions. |
| [Paper figures](examples/paper_figures.ipynb) | Plot image-selection fractions and compare the forecast with graded SuGOHI candidates. |

The cusp and quad calculation selects on third- and fourth-image brightness
and angular resolution, assuming subtracted lens light. For a double-source
forecast, the first survey defines the sky area. Each source has its own
photometry and detection limits. The returned count grid has axes
`(zs_near, zs_far, sigma, zl)`. Use its sum for the expected number of systems.
The five-dimensional tensor reports conditional weights for pairs in matching
magnitude bins.

`Compute_SL_distributions` saves the batch forecast under its survey title.
The original batch model uses SIE with light subtracted and SIS with lens
light included. Plot comparisons use an existing named result on the standard
grid. When they calculate a new SIE pair on that grid, they save it with the
suffix `SIE lens light`. Custom plotting grids are evaluated in memory.
The pickle format stores arrays. Keep the input grids and model choices
alongside any results you save manually.

## Performance and reproducibility

The survey integration evaluates source photometry and magnification bias
once per source plane. It reuses foreground galaxy counts and batches the
geometry for each foreground redshift. The magnitude sum retains its original
order. A timed run on macOS with Python 3.9.18 and the pinned dependencies
gave the following results for the Euclid VIS grid above.

| SIE forecast | Original main (`77a0960`) | Optimized | Expected lenses |
| --- | ---: | ---: | ---: |
| Lens light subtracted | 27.76 s | 0.267 s | 149,922 |
| Lens light included | 52.51 s | 1.508 s | 88,611 |

The comparison checks shapes, dtypes and every output byte, including all
four-dimensional magnitude bins and the six projections. From a Git checkout
with the original main commit available, run the regression suite with

```sh
python -m unittest discover -s tests -v
```

Set `GALESS_FULL_GRID=1` to include the timed full-grid comparison. The
smaller comparisons exercise both mass profiles, foreground light,
different bands, magnitude orderings and custom population functions.

### Corrections that affect results

- On original main, a worker limit could drop source planes, and parallel
  integration used the first redshift width throughout. Workers now cover
  every observable plane and use the same widths as the serial integral.
- The lens-magnitude histogram now assigns each positive lens redshift its
  own count slice when the grid starts at zero. Smoothed marginals also
  retain the requested length on two- and three-bin grids.
- In the dev double-source calculation, the farther source could overwrite
  the nearer source's photometry. The two selections are now independent.
  Empty conditional magnitude bins return zero. The source-magnitude
  histogram also applies its cut to the correct positive-redshift slices.
- The HOD model evaluates tabulated nonlinear power at each integration
  wavenumber. Effective Tinker bias uses peak height. Pass
  `USE_MY_BIAS=True` to reproduce dev's original squared-peak-height bias.

## Attribution

Please cite [Ferrami & Wyithe (2024)](https://arxiv.org/abs/2404.03143) when
using GaLeSS in a publication and link to the
[GaLeSS repository](https://github.com/Ferr013/GALESS).
Questions are welcome at [gferrami@student.unimelb.edu.au](mailto:gferrami@student.unimelb.edu.au).
