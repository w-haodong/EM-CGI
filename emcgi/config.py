"""Architecture defaults for the core release."""

def model_config(points=68, image_hw=(1024, 512)):
    cfg = {'schema': 'v132_silc_joint_residual_v2', 'method': 'SILC', 'image_hw': [1024, 512], 'independent_heatmaps': True, 'local_peak_radius': 4, 'structure_prior_scale': 0.03, 'hidden': 64, 'interaction_prior_scale': 1.0, 'relative_shift_limit_px': 16.0, 'noise_threshold': 0.1, 'confidence_threshold': 0.3, 'unary_covariance_floor_px': 1.0, 'edge_log_scale_limit': 0.5, 'search_std_min_px': 2.0, 'search_std_max_px': 48.0, 'search_radius_cap_px': 96.0, 'search_max_shift_px': 128.0, 'search_scale_factors': [1.0, 1.5, 2.0, 3.0], 'evidence_gate_bias': 0.0, 'write_gate_bias': 0.0, 'candidate_prior_bias': -2.0, 'candidate_prior_limit': 2.0, 'candidate_energy_cap': 64.0}
    cfg.update(points=int(points), image_hw=list(image_hw))
    return cfg
