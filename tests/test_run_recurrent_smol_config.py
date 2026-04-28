import inspect
import unittest

import run_recurrent_smol as smol


class JointHaltConfigTests(unittest.TestCase):
    def test_quality_and_slo_modes_resolve(self):
        cases = [
            ("dry_strathop_polish2_joint_halt_quality", 12.0, 0.08),
            ("dry_strathop_polish2_joint_halt_slo", 8.0, 0.10),
            ("dry_strathop_polish2_joint_halt_quality_agdistill", 12.0, 0.08),
            ("dry_strathop_polish2_joint_halt_stability", 6.0, 0.0),
            ("core3_8n4h_strathop_joint_halt_quality", 12.0, 0.08),
            ("core3_8n4h_strathop_joint_halt_quality_agdistill", 12.0, 0.08),
            ("core3_8n4h_strathop_joint_halt_quality_stability", 12.0, 0.08),
            ("core3_8n4h_strathop_joint_halt_slo", 8.0, 0.10),
            ("core3_8n4h_strathop_joint_halt_slo_stability", 8.0, 0.10),
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
                if "_agdistill" in mode:
                    self.assertGreater(cfg.joint_halt_agreement_distill_weight, 0.0)
                    self.assertGreater(cfg.joint_halt_agreement_route_weight, 0.0)
                if "_stability" in mode:
                    self.assertTrue(cfg.use_stability_head)
                    self.assertTrue(cfg.utility_use_stability_feature)

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
                "core3_8n4h_strathop_joint_halt_quality_agdistill_reuse_highval",
                "core3_8n4h_strathop_joint_halt_quality_agdistill_seed{seed}",
            ),
            (
                "core3_8n4h_strathop_polish2_joint_halt_quality_stability_reuse_highval",
                "core3_8n4h_strathop_polish2_joint_halt_quality_stability_seed{seed}",
            ),
            (
                "core3_8n4h_strathop_joint_halt_slo_stability_reuse_highval",
                "core3_8n4h_strathop_joint_halt_slo_stability_seed{seed}",
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

    def test_probe_audit_is_scoped_to_quality_stability_highval(self):
        probe_cfg = smol.config_for_mode(
            "core3_8n4h_strathop_joint_halt_quality_stability_reuse_highval"
        )
        self.assertTrue(probe_cfg.router_probe_audit)
        self.assertTrue(probe_cfg.router_selective_agree_audit)
        self.assertEqual(probe_cfg.joint_halt_steps, 0)
        self.assertEqual(probe_cfg.router_val_batches, 256)
        self.assertEqual(probe_cfg.eval_batches, 256)

        plain_quality_cfg = smol.config_for_mode(
            "core3_8n4h_strathop_joint_halt_quality_reuse_highval"
        )
        self.assertFalse(plain_quality_cfg.router_probe_audit)
        self.assertFalse(plain_quality_cfg.router_selective_agree_audit)

    def test_natgraph_bridge_modes_resolve(self):
        teacher_cfg = smol.config_for_mode("core3_8n4h_natgraph_teacher")
        self.assertEqual(teacher_cfg.prompt_style, "natural_graph")
        self.assertEqual(teacher_cfg.checkpoint_tag, "core3_8n4h_natgraph_seed{seed}")
        self.assertTrue(teacher_cfg.save_checkpoints)
        self.assertGreaterEqual(teacher_cfg.max_length, 224)

        train_cfg = smol.config_for_mode("core3_8n4h_natgraph_joint_halt_quality_stability")
        self.assertEqual(train_cfg.prompt_style, "natural_graph")
        self.assertTrue(train_cfg.use_utility_router)
        self.assertTrue(train_cfg.use_stability_head)
        self.assertEqual(train_cfg.load_checkpoint_tag, "core3_8n4h_natgraph_seed{seed}")
        self.assertEqual(
            train_cfg.checkpoint_tag,
            "core3_8n4h_natgraph_joint_halt_quality_stability_seed{seed}",
        )
        self.assertEqual(train_cfg.jump_steps, 4500)
        self.assertEqual(train_cfg.joint_halt_steps, 2000)
        self.assertFalse(train_cfg.load_jumprec_state)

        reuse_cfg = smol.config_for_mode(
            "core3_8n4h_natgraph_joint_halt_quality_stability_reuse_highval"
        )
        self.assertEqual(reuse_cfg.prompt_style, "natural_graph")
        self.assertEqual(reuse_cfg.joint_halt_steps, 0)
        self.assertFalse(reuse_cfg.save_checkpoints)
        self.assertTrue(reuse_cfg.load_jumprec_state)
        self.assertTrue(reuse_cfg.router_selective_agree_audit)
        self.assertEqual(reuse_cfg.router_val_batches, 256)

    def test_natgraph_prompt_formatter_keeps_answer_blank(self):
        labels = smol.route_label_names("natural_graph", 4)
        prompt = smol.format_route_prompt(
            "natural_graph",
            "alternate",
            [(0, 1), (1, 2), (2, 3), (3, 0)],
            labels,
            0,
            3,
        )
        self.assertIn("route card", prompt)
        self.assertIn("alternating each move", prompt)
        self.assertEqual(prompt.split("Answer:", 1)[1].strip(), "")

    def test_slo_modes_sample_route_operating_point(self):
        cfg = smol.config_for_mode("core3_8n4h_strathop_joint_halt_slo")
        self.assertGreater(cfg.joint_halt_false_accept_weight_max, cfg.utility_false_accept_weight)
        self.assertGreaterEqual(cfg.joint_halt_cost_weight_min, 0.0)
        self.assertGreater(cfg.joint_halt_cost_weight_max, cfg.joint_halt_cost_weight_min)

    def test_quality_cats_modes_resolve(self):
        train_cfg = smol.config_for_mode("core3_8n4h_strathop_joint_halt_quality_cats")
        self.assertTrue(train_cfg.use_utility_router)
        self.assertTrue(train_cfg.use_consistency_head)
        self.assertEqual(train_cfg.joint_halt_steps, 0)
        self.assertEqual(train_cfg.consistency_steps, 2000)
        self.assertEqual(
            train_cfg.load_checkpoint_tag,
            "core3_8n4h_strathop_joint_halt_quality_seed{seed}",
        )
        self.assertEqual(
            train_cfg.checkpoint_tag,
            "core3_8n4h_strathop_joint_halt_quality_cats_seed{seed}",
        )
        self.assertTrue(train_cfg.save_checkpoints)

        reuse_cfg = smol.config_for_mode("core3_8n4h_strathop_polish2_joint_halt_quality_cats_reuse_highval")
        self.assertTrue(reuse_cfg.use_consistency_head)
        self.assertEqual(reuse_cfg.consistency_steps, 0)
        self.assertFalse(reuse_cfg.save_checkpoints)
        self.assertEqual(
            reuse_cfg.load_checkpoint_tag,
            "core3_8n4h_strathop_polish2_joint_halt_quality_cats_seed{seed}",
        )
        self.assertEqual(reuse_cfg.router_val_batches, 256)
        self.assertEqual(reuse_cfg.eval_batches, 256)

    def test_dry_quality_cats_modes_resolve(self):
        train_cfg = smol.config_for_mode("dry_strathop_polish2_joint_halt_quality_cats")
        self.assertTrue(train_cfg.use_utility_router)
        self.assertTrue(train_cfg.use_consistency_head)
        self.assertEqual(train_cfg.consistency_steps, 4)
        self.assertEqual(
            train_cfg.load_checkpoint_tag,
            "dry_strathop_polish2_joint_halt_quality_seed{seed}",
        )
        self.assertEqual(
            train_cfg.checkpoint_tag,
            "dry_strathop_polish2_joint_halt_quality_cats_seed{seed}",
        )
        self.assertTrue(train_cfg.save_checkpoints)

        reuse_cfg = smol.config_for_mode("dry_strathop_polish2_joint_halt_quality_cats_reuse")
        self.assertTrue(reuse_cfg.use_consistency_head)
        self.assertEqual(reuse_cfg.consistency_steps, 0)
        self.assertFalse(reuse_cfg.save_checkpoints)
        self.assertEqual(
            reuse_cfg.load_checkpoint_tag,
            "dry_strathop_polish2_joint_halt_quality_cats_seed{seed}",
        )

    def test_agreement_aux_does_not_mark_final_budget_unsafe(self):
        source = inspect.getsource(smol.run_experiment)
        self.assertIn("def utility_router_agreement_bce", source)
        self.assertIn("def joint_halt_agreement_distill_loss", source)
        self.assertIn("def candidate_agreement_prob_stack", source)
        self.assertIn("def run_router_probe_audit", source)
        self.assertIn("probe_upper_bound", source)
        self.assertIn("def choose_selective_agree_thresholds", source)
        self.assertIn("selective_agreement", source)
        self.assertIn("serial_jumprec_selective_agree", source)
        self.assertIn("jumprec_serial_selective_agree_speed_ms_per_batch", source)
        self.assertIn("weights[-1] = 0.0", source)
        self.assertIn("stable_target[-1] = (pred_stack[-1] == full_pred).float()", source)

    def test_utility_controller_audit_policies_are_registered(self):
        source = inspect.getsource(smol.run_experiment)
        self.assertIn("utility_then_agree_floors", source)
        self.assertIn("def route_utility_then_agree_predictions", source)
        self.assertIn("agree_then_utility_floors", source)
        self.assertIn("def route_agree_then_utility_predictions", source)
        self.assertIn("choose_per_budget_utility_thresholds", source)
        self.assertIn("utility_per_budget", source)


if __name__ == "__main__":
    unittest.main()
