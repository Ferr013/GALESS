"""Compare the survey outputs with the published main implementation on this machine."""
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
import zipfile

import numpy as np
from galess.LensStat import lens_stat as ls
from galess.LensStat import parallel_ls
from galess.Utils import ls_utils as utils


MAIN = '77a09603361238eb545dccee4c4c3fd0ec6c7e13'


class SurveyComparison(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Use main's own source and SIE tables for the reference integration.
        repo = Path(__file__).resolve().parents[1]
        archive = subprocess.check_output([
            'git', 'archive', '--format=zip', MAIN, 'galess/LensStat/lens_stat.py',
            'galess/data/SIE_dPdmu'], cwd=repo)
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(io.BytesIO(archive)) as files:
                files.extractall(tmp)
            path = Path(tmp) / 'galess/LensStat/lens_stat.py'
            spec = importlib.util.spec_from_file_location('main_lens_stat', path)
            cls.reference = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.reference)
        cls.reference.tqdm = iter

    def compare(self, survey, sigma, zl, zs, M, **options):
        p = utils.read_survey_params(survey)
        args = (sigma, zl, zs, M, p['limit'], p['area'], p['seeing'], 20,
                p['exp_time_sec'], p['sky_bckgnd_m'], p['zero_point_m'], p['photo_band'])
        runs = []
        for module in (self.reference, ls):
            start = time.perf_counter()
            outputs = module.calculate_num_lenses_and_prob(*args, mag_cut=p['cut'], **options)
            elapsed = time.perf_counter() - start
            self.assertGreater(outputs[0].sum(), 0)
            derived = module.get_N_and_P_projections(outputs[0], sigma, zl, zs)
            runs.append((outputs + derived, elapsed))
        for old, new in zip(runs[0][0], runs[1][0]):
            self.assertEqual(old.shape, new.shape)
            self.assertEqual(old.dtype, new.dtype)
            self.assertTrue(old.tobytes() == new.tobytes(), 'Output differs from original main')
        return runs[0][1], runs[1][1]

    def test_survey_grids(self):
        sigma = np.array([118., 166., 245., 330.])
        zl = np.array([0., 0.12, 0.41, 0.95, 1.8])
        zs = np.array([0., 0.4, 0.9, 2.5, 5.2, 11.])
        M = np.linspace(-15, -25, 12)
        for sie in (False, True):
            for light in (False, True):
                with self.subTest(sie=sie, light=light):
                    self.compare('EUCLID Wide VIS', sigma, zl, zs, M,
                                 LENS_LIGHT_FLAG=light, SIE_FLAG=sie)
        # Exercise different photometry, magnitude ordering and population models.
        self.compare('COSMOS Web F150W', sigma, zl, zs, M[::-1],
                     Phi_vel_disp=ls.Phi_vel_disp_Geng, num_exposures=3,
                     LF_func=lambda M, z: 0.75 * ls.schechter_LF(M, z))
        self.compare('DES i band', sigma, zl, zs, M[::-1], Phi_vel_disp=ls.Phi_vel_disp_SDSS,
                     arc_mu_threshold=4, seeing_trsh=0.7, SIE_FLAG=False)

    def test_parallel_schedules_every_visible_plane(self):
        sigma = np.array([140., 250., 320.])
        zl = np.array([0., 0.18, 0.7])
        zs = np.array([0., 0.4, 0.95, 2., 11.])
        args = (sigma, zl, zs, np.linspace(-24., -16., 8), 26., 30.,
                0.13, 10, 1000., 21., 25., 'sdss_i0')
        evolution = (-1.1, 0.25, 0.05, -0.05)
        expected = self.reference.calculate_num_lenses_and_prob(
            *args, Phi_vel_disp=lambda s, z: ls.Phi_vel_disp_GENERIC(s, z, *evolution))[0]
        visible = zs <= ls.get_highest_LYA_rest_fram_observable('sdss_i0')
        for workers in (1, 2, 7):
            with self.subTest(workers=workers):
                actual = parallel_ls.calculate_num_lenses_and_prob(
                    *args, Phi_vel_disp=ls.Phi_vel_disp_GENERIC, VDF_args=evolution, ncores=workers)
                self.assertEqual(actual.shape, expected[visible].shape)
                self.assertEqual(actual.tobytes(), expected[visible].tobytes())

    @unittest.skipUnless(os.environ.get('GALESS_FULL_GRID'), 'set GALESS_FULL_GRID=1')
    def test_euclid_full_grid(self):
        for light in (False, True):
            old, new = self.compare('EUCLID Wide VIS', np.linspace(100, 400, 31),
                                    np.arange(0, 2.5, 0.1), np.arange(0, 5.4, 0.2),
                                    np.linspace(-13, -25, 25), LENS_LIGHT_FLAG=light)
            print(f'Euclid VIS (lens light {light})  main {old:.3f}s  branch {new:.3f}s')
