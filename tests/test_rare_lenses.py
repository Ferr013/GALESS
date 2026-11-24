"""Independent checks for double-source selection, histogram binning and saved results."""

import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from galess.LensStat import lens_stat as ls
from galess.RareLensStat import rare_lens_stat as rare
from galess.Utils import ls_utils as utils


class RareLensTests(unittest.TestCase):
    def test_double_source_count_from_independent_single_surveys(self):
        sigma = np.array([155., 220., 310.])
        zl = np.array([0., .2, .4])
        zs = np.array([0., .65, 1.5, 2.8])
        mags = np.linspace(-16., -24., 8)
        grid = (sigma, zl, zs, mags)
        # The second survey follows up all selected lenses in the first area.
        near = (25., 40., .25, 15., 100., 20.5, 26., 'sdss_i0')
        far = (26., 40., .25, 10., 200., 21., 27.5, 'sdss_g0')
        model = dict(LENS_LIGHT_FLAG=True, SIE_FLAG=True)
        counts, r1, r2, tensor = rare.calculate_num_double_lenses_and_prob(
            *grid, *near, 24., *far, **model)
        near_counts, near_radius, _ = ls.calculate_num_lenses_and_prob(
            *grid, *near, mag_cut=24., **model)
        far_counts, far_radius, _ = ls.calculate_num_lenses_and_prob(*grid, *far, **model)

        # Each single-survey count already includes the foreground galaxies.
        # Counting two sources behind one lens removes that factor once.
        self.assertGreater(counts.sum(), 0)
        self.assertTrue(np.isfinite(tensor).all())
        for i, s in enumerate(sigma):
            ds = sigma[max(i, 1)] - sigma[max(i, 1)-1]
            for j, z in enumerate(zl):
                if z == 0:
                    self.assertFalse(counts[:, :, i, j].any())
                    continue
                dz = zl[j] - zl[j-1]
                density = ls.Phi_vel_disp_Mason(s - ds/2, z) + ls.Phi_vel_disp_Mason(s + ds/2, z)
                ngal = ls.Lens_cone_volume_diff(z, near[1], dz) * density * ds/2
                independent = near_counts[:, i, j, None] * far_counts[:, i, j] / ngal
                np.testing.assert_allclose(counts[:, :, i, j], np.triu(independent, k=1),
                    rtol=1e-14, atol=0)
        np.testing.assert_array_equal(r1[r1 > 0], near_radius[r1 > 0])
        np.testing.assert_array_equal(r2[r2 > 0], far_radius[r2 > 0])

    def test_magnitude_cut_and_positive_redshift_slices(self):
        zs = np.array([0., .5, 2.])
        mags = np.array([-23., -20., -17.])
        m_obs = np.arange(15., 36.)
        counts = np.array([900., 1., 10.])[:, None, None, None] * np.ones(3)
        expected = np.zeros(len(m_obs))
        for iz in (1, 2):
            z = zs[iz]
            dm = 5 * np.log10(ls.cosmo_luminosity_distance(z) * 1e5)
            for im, mag in enumerate(mags):
                corr = ls.K_correction_from_UV(z, 'sdss_i0', mag)
                m = mag + (dm + corr) - 2.5 * np.log10(3)
                if m <= 24.:
                    expected[np.argmin(abs(m_obs - m))] += counts[iz, 0, 0, im]
        self.assertTrue(0 < expected.sum() < 33)
        actual = rare.get_src_magnitude_distr(m_obs, 24., zs, counts, mags[::-1])
        np.testing.assert_array_equal(actual, expected)

    def test_double_source_pickle_layout(self):
        matrix, r1, r2, tensor = (np.arange(n) for n in (2, 3, 4, 5))
        with tempfile.TemporaryDirectory() as root, patch.object(utils, 'BASEPATH', root):
            utils.save_pickled_files('Test Survey', matrix, (r1, r2), tensor,
                matrix + 4, (r1 + 4, r2 + 4), tensor + 4, DOUBLE_LENS=True)
            values = utils.load_pickled_files('Test Survey', DOUBLE_LENS=True)
            self.assertEqual(len(values), 8)
            for index, result in enumerate((matrix, r1, r2, tensor)):
                np.testing.assert_array_equal(values[index], result)
                np.testing.assert_array_equal(values[index + 4], result + 4)
