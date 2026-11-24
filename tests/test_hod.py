"""Numerical regressions for the halo occupation model added on dev."""
import unittest

import numpy as np
from scipy.integrate import quad

from galess.HOD import HOD


class HaloOccupation(unittest.TestCase):
    def test_population_at_z_1_7(self):
        # Reference values from dev 44ea517, using the shipped mass function.
        model = (10.**12.46, 0.2, 10.**14.18, 1., 1.7)
        actual = [HOD.get_AVG_N_tot(*model), HOD.get_AVG_Host_Halo_Mass(*model),
            HOD.get_AVG_f_sat(*model), HOD.get_EFF_gal_bias(*model, USE_MY_BIAS=True),
            HOD.get_EFF_gal_bias(*model)]
        np.testing.assert_allclose(actual, [0.00038965661893712453,
            6379136201194.083, 0.038239405751852, 12.357237000998612,
            2.5787203495589006], rtol=2e-14, atol=0)

        mass = HOD.init_lookup_table(1.7)[0]
        mass[:] = 0
        self.assertTrue((HOD.init_lookup_table(1.7)[0] > 0).all())

    def test_tabulated_two_halo_projection(self):
        # Dev formerly multiplied the entire power array at each scalar quad node.
        # Integrate each segment of a piecewise-linear test spectrum directly.
        model = (10.**12.46, 0.2, 10.**14.18, 1.)
        z, theta = 1.7, 1./206265
        mass, hmf, nu, _, _ = HOD.init_lookup_table(z)
        k = np.array([0.01, 0.1, 1.])
        power = np.array([4., 2., 0.5])
        density = HOD.n_g(*model, z, mass, hmf)
        central = HOD.N_cen(mass, model[0], model[1])
        satellite = HOD.N_sat(mass, model[2], model[3], model[0], model[1])
        bias = HOD.halo_bias_TINKER(np.sqrt(nu))
        crit = 4/3*np.pi*HOD.cosmo.critical_density(z).value*200*2e40
        distance = HOD.cosmo.comoving_distance(z).value
        args = (crit, mass, hmf, density, central, satellite, bias, k, power)

        def integrand(x):
            two = HOD.PS_2h(x, *model, z, *args, None, None, False)
            return np.interp(x, k, power)*two*HOD.factor_k(x, theta, distance)

        expected = sum(quad(integrand, a, b)[0] for a, b in zip(k[:-1], k[1:]))
        actual = HOD.omega_inner_integral_2(theta, *model, z, distance,
            *args, None, None, False)
        self.assertAlmostEqual(actual, expected, delta=1e-4)
