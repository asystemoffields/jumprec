import inspect
import unittest

import run_recurrent_smol as smol


class JointHaltConfigTests(unittest.TestCase):
    def test_quality_and_slo_modes_resolve(self):
        cases = [
            ("dry_strathop_polish2_joint_halt_quality", 12.0, 0.08),
            ("dry_strathop_polish2_joint_halt_slo", 8.0, 0.10),
            ("core3_8n4h_strathop_joint_halt_quality", 12.0, 0.08),
            ("core3_8n4h_strathop_joint_halt_slo", 8.0, 0.10),
            ("core3_8n4h_strathop_polish2_joint_halt_quality_reuse_highval", 12.0, 0.08),
            ("core3_8n4h_strathop_polish2_joint_halt_slo_reuse_highval", 8.0, 0.10),
        ]
        for mode, false_accept_weight, agreement_weight in cases:
            with self.subTest(mode=mode):
                cfg = smol.config_for_mode(mode)
                self.assertTrue(cfg.use_utility_router)
                self.assertEqual(cfg.utility_false_accept_weight, false_accept_weight)
                self.assertEqual(cfg.joint_halt_agreement_bce_weight, agreement_weight)
                self.assertGreater(cfg.joint_halt_candidate_ce_weight, 0.0)
                self.assertGreater(cfg.joint_halt_candidate_distill_weight, 0.0)

    def test_reuse_modes_load_matching_objective_family(self):
        cases = [
            (
                "core3_8n4h_strathop_joint_halt_quality_reuse_highval",
                "core3_8n4h_strathop_joint_halt_quality_seed{seed}",
            ),
            (
                "core3_8n4h_strathop_joint_halt_slo_reuse_highval",
                "core3_8n4h_strathop_joint_halt_slo_seed{seed}",
            ),
            (
                "core3_8n4h_strathop_polish2_joint_halt_quality_stability_reuse_highval",
                "core3_8n4h_strathop_polish2_joint_halt_quality_stability_seed{seed}",
            ),
        ]
        for mode, load_tag in cases:
            with self.subTest(mode=mode):
                cfg = smol.config_for_mode(mode)
                self.assertEqual(cfg.joint_halt_steps, 0)
                self.assertFalse(cfg.save_checkpoints)
                self.assertEqual(cfg.load_checkpoint_tag, load_tag)
                self.assertEqual(cfg.router_val_batches, 256)
                self.assertEqual(cfg.eval_batches, 256)

    def test_slo_modes_sample_route_operating_point(self):
        cfg = smol.config_for_mode("core3_8n4h_strathop_joint_halt_slo")
        self.assertGreater(cfg.joint_halt_false_accept_weight_max, cfg.utility_false_accept_weight)
        self.assertGreaterEqual(cfg.joint_halt_cost_weight_min, 0.0)
        self.assertGreater(cfg.joint_halt_cost_weight_max, cfg.joint_halt_cost_weight_min)

    def test_agreement_aux_does_not_mark_final_budget_unsafe(self):
        source = inspect.getsource(smol.run_experiment)
        self.assertIn("def utility_router_agreement_bce", source)
        self.assertIn("weights[-1] = 0.0", source)


if __name__ == "__main__":
    unittest.main()
