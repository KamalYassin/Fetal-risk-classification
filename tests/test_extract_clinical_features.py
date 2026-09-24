"""Focused regression tests for CTG preprocessing and event continuity."""

import sys
import unittest
from pathlib import Path

import numpy as np

FEATURE_DIR = Path(__file__).resolve().parents[1] / "src" / "features"
sys.path.insert(0, str(FEATURE_DIR))

import extract_clinical_features as clinical


class FhrPreprocessingTests(unittest.TestCase):
    def test_raw_signal_is_preserved_and_artifacts_become_nan(self):
        raw = np.array([140.0, 141.0, 180.0, 140.0, 240.0, 0.0])
        result = clinical.preprocess_fhr(raw)

        np.testing.assert_array_equal(result["raw_fhr"], raw)
        self.assertTrue(np.isnan(result["cleaned_fhr"][2]))
        self.assertTrue(np.isnan(result["cleaned_fhr"][4]))
        self.assertTrue(np.isnan(result["cleaned_fhr"][5]))
        self.assertTrue(result["artifact_mask"][2])
        self.assertTrue(result["artifact_mask"][4])
        self.assertFalse(result["artifact_mask"][5])

    def test_nan_gap_breaks_an_event(self):
        signal = np.r_[np.full(10, 160.0), np.nan, np.full(10, 160.0)]
        baseline = np.full(len(signal), 140.0)
        result = clinical.detect_accelerations(signal, baseline, 1.0)
        self.assertEqual(result["count"], 0)

        signal = np.r_[np.full(15, 160.0), np.nan, np.full(14, 160.0)]
        result = clinical.detect_accelerations(signal, baseline=np.full(len(signal), 140.0), sampling_frequency=1.0)
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["events"][0]["start_index"], 0)
        self.assertEqual(result["events"][0]["end_index"], 15)
        self.assertEqual(result["events"][0]["duration_seconds"], 15.0)
        self.assertEqual(result["events"][0]["amplitude_or_depth"], 20.0)


class BaselineAndVariabilityTests(unittest.TestCase):
    def test_moving_baseline_follows_a_long_level_shift(self):
        fhr = np.r_[np.full(600, 120.0), np.full(600, 160.0)]
        baseline = clinical.estimate_baseline_series(fhr, 1.0)
        self.assertAlmostEqual(baseline[300], 120.0)
        self.assertAlmostEqual(baseline[900], 160.0)

    def test_baseline_is_nan_in_an_insufficient_valid_window(self):
        fhr = np.full(600, np.nan)
        fhr[300] = 140.0
        baseline = clinical.estimate_baseline_series(fhr, 1.0)
        self.assertTrue(np.isnan(baseline[300]))

    def test_ltv_is_not_duplicate_signal_standard_deviation(self):
        fhr = np.tile([130.0, 150.0], 60)
        ltv = clinical.calculate_long_term_variability(fhr, 1.0)
        self.assertAlmostEqual(ltv, 0.0)
        self.assertGreater(np.std(fhr), ltv)


class ContractionAndQualityTests(unittest.TestCase):
    def test_narrow_uc_spike_is_rejected_but_broad_peak_is_retained(self):
        uc = np.zeros(500)
        uc[100] = 80.0
        uc[250:281] = np.linspace(0.0, 50.0, 31)
        uc[281:312] = np.linspace(50.0, 0.0, 31)
        result = clinical.detect_contractions(uc, 1.0)
        self.assertEqual(len(result["events"]), 1)
        self.assertGreaterEqual(result["events"][0]["duration_seconds"], 20.0)

    def test_longest_missing_gap_counts_contiguous_samples(self):
        mask = [False, True, True, False, True, True, True, False]
        self.assertEqual(clinical.longest_true_run(mask), 3)


class AdditionalFeatureTests(unittest.TestCase):
    def test_bradycardia_and_tachycardia_use_valid_fhr_denominator(self):
        fhr = np.array([100.0, 109.0, 110.0, 160.0, 161.0, np.nan])
        result = clinical.calculate_fhr_threshold_exposure(fhr, 2.0)
        self.assertEqual(result["bradycardia_duration_seconds"], 1.0)
        self.assertEqual(result["bradycardia_percentage"], 40.0)
        self.assertEqual(result["tachycardia_duration_seconds"], 0.5)
        self.assertEqual(result["tachycardia_percentage"], 20.0)

    def test_contraction_deceleration_interaction_uses_post_peak_window(self):
        contractions = [{"peak_index": 100}, {"peak_index": 300}]
        decelerations = [{"start_index": 50}, {"start_index": 150}, {"start_index": 450}]
        result = clinical.calculate_contraction_deceleration_interactions(
            contractions, decelerations, sampling_frequency=1.0
        )
        self.assertEqual(
            result["contractions_followed_by_deceleration_percentage"], 50.0
        )
        self.assertEqual(
            result["mean_delay_contraction_to_deceleration_seconds"], 50.0
        )

    def test_baseline_crossings_reset_at_missing_samples(self):
        fhr = np.array([138.0, 142.0, 140.0, 138.0, np.nan, 142.0, 138.0])
        baseline = np.full(len(fhr), 140.0)
        self.assertEqual(clinical.calculate_baseline_crossing_count(fhr, baseline), 3)

    def test_slope_features_use_only_adjacent_valid_samples(self):
        fhr = np.array([100.0, 102.0, 101.0, np.nan, 90.0, 87.0])
        result = clinical.calculate_fhr_slope_features(fhr, 1.0)
        self.assertEqual(result["mean_positive_fhr_slope"], 2.0)
        self.assertEqual(result["mean_negative_fhr_slope"], -2.0)
        self.assertEqual(result["maximum_negative_fhr_slope"], -3.0)

    def test_segment_variability_uses_complete_five_minute_windows(self):
        one_segment = np.tile([130.0, 150.0], 150)
        fhr = np.r_[one_segment, one_segment, np.full(20, 200.0)]
        result = clinical.calculate_segment_variability_features(fhr, 1.0)
        self.assertAlmostEqual(result["segment_stv_mean"], 20.0)
        self.assertAlmostEqual(result["segment_stv_std"], 0.0)
        self.assertAlmostEqual(result["segment_ltv_mean"], 0.0)
        self.assertAlmostEqual(result["segment_ltv_std"], 0.0)


if __name__ == "__main__":
    unittest.main()
