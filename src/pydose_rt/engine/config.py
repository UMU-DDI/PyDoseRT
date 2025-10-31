import numpy as np

class config:
    comet_project_name = "autoplan"
    comet_workspace = "vuhoangminh"
    input_shape_attila = (128, 128, 320, 8)
    input_shape_over_under_prediction = (128, 128, 320, 11)
    input_shape = (128, 128, 320, 6)  # including background
    ct_array_shape = (128, 128, 320)

    dir_generated_samples_from_dose_network = (
        "database/generated_samples_from_dose_network"
    )
    dir_generated_samples_from_dose_network_laplace = (
        "/data/minh/github/autoplan/database/generated_samples_from_dose_network"
    )

    print_dict = {
        "total_loss": "total",
        "loss_lower_bound_gy": "l.gy",
        "loss_higher_bound_gy": "h.gy",
        "loss_lower_bound_target": "l.tar",
        "loss_higher_bound_target": "h.tar",
        "l2_loss_oars_and_background": "l2",
        "aux_loss": "aux",
        "mu_rate_loss": "mu.r",
        "mu_complexity_loss": "mu.c",
        "leaf_opening_loss": "leaf.o",
        "leaf_rate_loss": "leaf.r",
    }

    structure_names = {
        "PTV": "PTV",
        "ROI1": "PenileBulb",
        "ROI2": "FemoralHead_L",
        "ROI3": "FemoralHead_R",
        "ROI4": "Bladder",
        "ROI5": "Rectum",
        "ROI6": "Background",
    }

    roi_colors = {
        "PTV": "black",
        "ROI1": "red",
        "ROI2": "green",
        "ROI3": "blue",
        "ROI4": "purple",
        "ROI5": "brown",
        "ROI6": "orange",
    }

    constraints_lund_probe = {
        "lower_bound_gy": {
            "PTV": 42.7,
            "ROI1": 0,    # PenileBulb
            "ROI2": 0,    # FemoralHead_L
            "ROI3": 0,    # FemoralHead_R
            "ROI4": 0,    # Bladder
            "ROI5": 0,    # Rectum
            "ROI6": 0,    # Background
        },
        "higher_bound_gy": {
            "PTV": 42.7, 
            "ROI1": 40.4, 
            "ROI2": 28.9, 
            "ROI3": 28.9, 
            "ROI4": 40.4, 
            "ROI5": 40.4, 
            "ROI6": 40.4, 
        },
        "lower_bound_target_percent": {
            "PTV": 95,    # ≥95% of PTV should get ≥42.7 Gy
            "ROI1": 0,
            "ROI2": 0,
            "ROI3": 0,
            "ROI4": 0,
            "ROI5": 0,
            "ROI6": 0,
        },
        "higher_bound_target_percent": {
            "PTV": 100,
            "ROI1": 100,  # PenileBulb
            "ROI2": 100,  # FemoralHead_L max limited below 28.9 Gy (scaled from <50 Gy)
            "ROI3": 100,  # FemoralHead_R max limited below 28.9 Gy
            "ROI4": 75,   # Bladder V40.4 < 25% (scaled from V70 < 25%)
            "ROI5": 85,   # Rectum V40.4 < 15% (scaled from V70 < 15%)
            "ROI6": 96,   # ≤4% around PTV allowed above background limit
        },
        "weight": {
            "PTV": 1000,
            "ROI1": 0.1,
            "ROI2": 0.1,
            "ROI3": 0.1,
            "ROI4": 0.1,
            "ROI5": 100,
            "ROI6": 1,
        },
    }

    constraints = {
        "lower_bound_gy": {
            "PTV": 74,
            "ROI1": 0,  # PenileBulb
            "ROI2": 0,  # FemoralHead_L
            "ROI3": 0,  # FemoralHead_R
            "ROI4": 0,  # Bladder
            "ROI5": 0,  # Rectum
            "ROI6": 0,  # Background
        },
        "higher_bound_gy": {
            "PTV": 83,
            "ROI1": 70,  # PenileBulb
            "ROI2": 50,  # FemoralHead_L
            "ROI3": 50,  # FemoralHead_R
            "ROI4": 70,  # Bladder
            "ROI5": 70,  # Rectum
            "ROI6": 70,  # Background
        },
        "lower_bound_target_percent": {
            "PTV": 95,  # Target: ≥95% of PTV should get at least 74 Gy, hotspots ≤83 Gy
            "ROI1": 0,  # PenileBulb
            "ROI2": 0,  # FemoralHead_L
            "ROI3": 0,  # FemoralHead_R
            "ROI4": 0,  # Bladder
            "ROI5": 0,  # Rectum
            "ROI6": 0,  # Background
        },
        "higher_bound_target_percent": {
            "PTV": 100,
            "ROI1": 100,  # PenileBulb
            "ROI2": 100,  # FemoralHead_L Maximum dose limited to <50 Gy
            "ROI3": 100,  # FemoralHead_R Maximum dose limited to <50 Gy
            "ROI4": 75,  # Bladder V70 < 25% (most voxels should be below 70 Gy)
            "ROI5": 85,  # Rectum V70 < 15% (most voxels should be below 70 Gy)
            "ROI6": 96,  # below 4 percent: around ptv
        },
        "weight": {
            "PTV": 100,
            "ROI1": 1,
            "ROI2": 2,
            "ROI3": 2,
            "ROI4": 2,
            "ROI5": 2,
            "ROI6": 1,
            # add tissue penalty, separate background and tissue
        },
    }

    # Original. PTV is target
    constraints_1 = {
        "lower_bound_gy": {
            "PTV": 74,
            "ROI1": 0,
            "ROI2": 0,
            "ROI3": 0,
            "ROI4": 0,
            "ROI5": 0,
            "ROI6": 0,
        },
        "higher_bound_gy": {
            "PTV": 83,
            "ROI1": 70,
            "ROI2": 50,
            "ROI3": 50,
            "ROI4": 70,
            "ROI5": 70,
            "ROI6": 70,
        },
        "lower_bound_target_percent": {
            "PTV": 95,
            "ROI1": 0,
            "ROI2": 0,
            "ROI3": 0,
            "ROI4": 0,
            "ROI5": 0,
            "ROI6": 0,
        },
        "higher_bound_target_percent": {
            "PTV": 100,
            "ROI1": 100,
            "ROI2": 100,
            "ROI3": 100,
            "ROI4": 75,
            "ROI5": 85,
            "ROI6": 97,
        },
        "weight": {
            "PTV": 100,
            "ROI1": 1,
            "ROI2": 2,
            "ROI3": 2,
            "ROI4": 2,
            "ROI5": 2,
            "ROI6": 1,
        },
    }

    # FemoralHead_L is target
    constraints_2 = {
        "lower_bound_gy": {
            "PTV": 0,
            "ROI1": 0,
            "ROI2": 74,
            "ROI3": 0,
            "ROI4": 0,
            "ROI5": 0,
            "ROI6": 0,
        },
        "higher_bound_gy": {
            "PTV": 70,
            "ROI1": 70,
            "ROI2": 83,
            "ROI3": 50,
            "ROI4": 70,
            "ROI5": 70,
            "ROI6": 70,
        },
        "lower_bound_target_percent": {
            "PTV": 0,
            "ROI1": 0,
            "ROI2": 95,
            "ROI3": 0,
            "ROI4": 0,
            "ROI5": 0,
            "ROI6": 0,
        },
        "higher_bound_target_percent": {
            "PTV": 75,
            "ROI1": 100,
            "ROI2": 100,
            "ROI3": 100,
            "ROI4": 75,
            "ROI5": 85,
            "ROI6": 97,
        },
        "weight": {
            "PTV": 2,
            "ROI1": 1,
            "ROI2": 100,
            "ROI3": 2,
            "ROI4": 2,
            "ROI5": 2,
            "ROI6": 1,
        },
    }
    # FemoralHead_R is target
    constraints_3 = {
        "lower_bound_gy": {
            "PTV": 0,
            "ROI1": 0,
            "ROI2": 0,
            "ROI3": 74,
            "ROI4": 0,
            "ROI5": 0,
            "ROI6": 0,
        },
        "higher_bound_gy": {
            "PTV": 70,
            "ROI1": 70,
            "ROI2": 50,
            "ROI3": 83,
            "ROI4": 70,
            "ROI5": 70,
            "ROI6": 70,
        },
        "lower_bound_target_percent": {
            "PTV": 0,
            "ROI1": 0,
            "ROI2": 0,
            "ROI3": 95,
            "ROI4": 0,
            "ROI5": 0,
            "ROI6": 0,
        },
        "higher_bound_target_percent": {
            "PTV": 75,
            "ROI1": 100,
            "ROI2": 100,
            "ROI3": 100,
            "ROI4": 75,
            "ROI5": 85,
            "ROI6": 97,
        },
        "weight": {
            "PTV": 2,
            "ROI1": 1,
            "ROI2": 2,
            "ROI3": 100,
            "ROI4": 2,
            "ROI5": 2,
            "ROI6": 1,
        },
    }
    # Both FemoralHead_L and FemoralHead_R are targets
    constraints_4 = {
        "lower_bound_gy": {
            "PTV": 0,
            "ROI1": 0,
            "ROI2": 74,
            "ROI3": 74,
            "ROI4": 0,
            "ROI5": 0,
            "ROI6": 0,
        },
        "higher_bound_gy": {
            "PTV": 70,
            "ROI1": 70,
            "ROI2": 83,
            "ROI3": 83,
            "ROI4": 70,
            "ROI5": 70,
            "ROI6": 70,
        },
        "lower_bound_target_percent": {
            "PTV": 0,
            "ROI1": 0,
            "ROI2": 95,
            "ROI3": 95,
            "ROI4": 0,
            "ROI5": 0,
            "ROI6": 0,
        },
        "higher_bound_target_percent": {
            "PTV": 75,
            "ROI1": 100,
            "ROI2": 100,
            "ROI3": 100,
            "ROI4": 75,
            "ROI5": 85,
            "ROI6": 97,
        },
        "weight": {
            "PTV": 2,
            "ROI1": 1,
            "ROI2": 100,
            "ROI3": 100,
            "ROI4": 2,
            "ROI5": 2,
            "ROI6": 1,
        },
    }
    # Rectum is target
    constraints_5 = {
        "lower_bound_gy": {
            "PTV": 0,
            "ROI1": 0,
            "ROI2": 0,
            "ROI3": 0,
            "ROI4": 0,
            "ROI5": 74,
            "ROI6": 0,
        },
        "higher_bound_gy": {
            "PTV": 70,
            "ROI1": 70,
            "ROI2": 50,
            "ROI3": 50,
            "ROI4": 70,
            "ROI5": 83,
            "ROI6": 70,
        },
        "lower_bound_target_percent": {
            "PTV": 0,
            "ROI1": 0,
            "ROI2": 0,
            "ROI3": 0,
            "ROI4": 0,
            "ROI5": 95,
            "ROI6": 0,
        },
        "higher_bound_target_percent": {
            "PTV": 85,
            "ROI1": 100,
            "ROI2": 100,
            "ROI3": 100,
            "ROI4": 75,
            "ROI5": 100,
            "ROI6": 97,
        },
        "weight": {
            "PTV": 2,
            "ROI1": 1,
            "ROI2": 2,
            "ROI3": 2,
            "ROI4": 2,
            "ROI5": 100,
            "ROI6": 1,
        },
    }
    # PTV and FemoralHead_L are targets
    constraints_6 = {
        "lower_bound_gy": {
            "PTV": 74,
            "ROI1": 0,
            "ROI2": 74,
            "ROI3": 0,
            "ROI4": 0,
            "ROI5": 0,
            "ROI6": 0,
        },
        "higher_bound_gy": {
            "PTV": 83,
            "ROI1": 70,
            "ROI2": 83,
            "ROI3": 50,
            "ROI4": 70,
            "ROI5": 70,
            "ROI6": 70,
        },
        "lower_bound_target_percent": {
            "PTV": 95,
            "ROI1": 0,
            "ROI2": 95,
            "ROI3": 0,
            "ROI4": 0,
            "ROI5": 0,
            "ROI6": 0,
        },
        "higher_bound_target_percent": {
            "PTV": 100,
            "ROI1": 100,
            "ROI2": 100,
            "ROI3": 100,
            "ROI4": 75,
            "ROI5": 85,
            "ROI6": 97,
        },
        "weight": {
            "PTV": 100,
            "ROI1": 1,
            "ROI2": 100,
            "ROI3": 2,
            "ROI4": 2,
            "ROI5": 2,
            "ROI6": 1,
        },
    }
    # Bladder is target
    constraints_7 = {
        "lower_bound_gy": {
            "PTV": 0,
            "ROI1": 0,
            "ROI2": 0,
            "ROI3": 0,
            "ROI4": 74,
            "ROI5": 0,
            "ROI6": 0,
        },
        "higher_bound_gy": {
            "PTV": 70,
            "ROI1": 70,
            "ROI2": 50,
            "ROI3": 50,
            "ROI4": 83,
            "ROI5": 70,
            "ROI6": 70,
        },
        "lower_bound_target_percent": {
            "PTV": 0,
            "ROI1": 0,
            "ROI2": 0,
            "ROI3": 0,
            "ROI4": 95,
            "ROI5": 0,
            "ROI6": 0,
        },
        "higher_bound_target_percent": {
            "PTV": 75,
            "ROI1": 100,
            "ROI2": 100,
            "ROI3": 100,
            "ROI4": 100,
            "ROI5": 85,
            "ROI6": 97,
        },
        "weight": {
            "PTV": 2,
            "ROI1": 1,
            "ROI2": 2,
            "ROI3": 2,
            "ROI4": 100,
            "ROI5": 2,
            "ROI6": 1,
        },
    }
    # No targets.
    constraints_8 = {
        "lower_bound_gy": {
            "PTV": 0,
            "ROI1": 0,
            "ROI2": 0,
            "ROI3": 0,
            "ROI4": 0,
            "ROI5": 0,
            "ROI6": 0,
        },
        "higher_bound_gy": {
            "PTV": 70,
            "ROI1": 70,
            "ROI2": 50,
            "ROI3": 50,
            "ROI4": 70,
            "ROI5": 70,
            "ROI6": 70,
        },
        "lower_bound_target_percent": {
            "PTV": 0,
            "ROI1": 0,
            "ROI2": 0,
            "ROI3": 0,
            "ROI4": 0,
            "ROI5": 0,
            "ROI6": 0,
        },
        "higher_bound_target_percent": {
            "PTV": 75,
            "ROI1": 100,
            "ROI2": 100,
            "ROI3": 100,
            "ROI4": 100,
            "ROI5": 85,
            "ROI6": 97,
        },
        "weight": {
            "PTV": 2,
            "ROI1": 1,
            "ROI2": 2,
            "ROI3": 2,
            "ROI4": 2,
            "ROI5": 2,
            "ROI6": 1,
        },
    }
    list_constraints = [
        constraints_1,
        constraints_2,
        constraints_3,
        constraints_4,
        constraints_5,
        constraints_6,
        constraints_7,
        constraints_8,
    ]

    constraints_w1 = {
        "lower_bound_gy": {
            "PTV": 74,
            "ROI1": 0,
            "ROI2": 0,
            "ROI3": 0,
            "ROI4": 0,
            "ROI5": 0,
            "ROI6": 0,
        },
        "higher_bound_gy": {
            "PTV": 83,
            "ROI1": 70,
            "ROI2": 50,
            "ROI3": 50,
            "ROI4": 70,
            "ROI5": 70,
            "ROI6": 70,
        },
        "lower_bound_target_percent": {
            "PTV": 95,
            "ROI1": 0,
            "ROI2": 0,
            "ROI3": 0,
            "ROI4": 0,
            "ROI5": 0,
            "ROI6": 0,
        },
        "higher_bound_target_percent": {
            "PTV": 100,
            "ROI1": 100,
            "ROI2": 100,
            "ROI3": 100,
            "ROI4": 75,
            "ROI5": 85,
            "ROI6": 96,
        },
        "weight": {
            "PTV": 100,
            "ROI1": 1,
            "ROI2": 2,
            "ROI3": 2,
            "ROI4": 2,
            "ROI5": 2,
            "ROI6": 1,
        },
    }

    constraints_w2 = {
        "lower_bound_gy": {
            "PTV": 74,
            "ROI1": 0,
            "ROI2": 0,
            "ROI3": 0,
            "ROI4": 0,
            "ROI5": 0,
            "ROI6": 0,
        },
        "higher_bound_gy": {
            "PTV": 83,
            "ROI1": 70,
            "ROI2": 50,
            "ROI3": 50,
            "ROI4": 70,
            "ROI5": 70,
            "ROI6": 70,
        },
        "lower_bound_target_percent": {
            "PTV": 95,
            "ROI1": 0,
            "ROI2": 0,
            "ROI3": 0,
            "ROI4": 0,
            "ROI5": 0,
            "ROI6": 0,
        },
        "higher_bound_target_percent": {
            "PTV": 100,
            "ROI1": 100,
            "ROI2": 100,
            "ROI3": 100,
            "ROI4": 75,
            "ROI5": 85,
            "ROI6": 96,
        },
        "weight": {
            "PTV": 10,
            "ROI1": 100,
            "ROI2": 2,
            "ROI3": 2,
            "ROI4": 2,
            "ROI5": 2,
            "ROI6": 1,
        },
    }

    constraints_w3 = {
        "lower_bound_gy": {
            "PTV": 74,
            "ROI1": 0,
            "ROI2": 0,
            "ROI3": 0,
            "ROI4": 0,
            "ROI5": 0,
            "ROI6": 0,
        },
        "higher_bound_gy": {
            "PTV": 83,
            "ROI1": 70,
            "ROI2": 50,
            "ROI3": 50,
            "ROI4": 70,
            "ROI5": 70,
            "ROI6": 70,
        },
        "lower_bound_target_percent": {
            "PTV": 95,
            "ROI1": 0,
            "ROI2": 0,
            "ROI3": 0,
            "ROI4": 0,
            "ROI5": 0,
            "ROI6": 0,
        },
        "higher_bound_target_percent": {
            "PTV": 100,
            "ROI1": 100,
            "ROI2": 100,
            "ROI3": 100,
            "ROI4": 75,
            "ROI5": 85,
            "ROI6": 96,
        },
        "weight": {
            "PTV": 10,
            "ROI1": 1,
            "ROI2": 100,
            "ROI3": 2,
            "ROI4": 2,
            "ROI5": 2,
            "ROI6": 1,
        },
    }

    constraints_w4 = {
        "lower_bound_gy": {
            "PTV": 74,
            "ROI1": 0,
            "ROI2": 0,
            "ROI3": 0,
            "ROI4": 0,
            "ROI5": 0,
            "ROI6": 0,
        },
        "higher_bound_gy": {
            "PTV": 83,
            "ROI1": 70,
            "ROI2": 50,
            "ROI3": 50,
            "ROI4": 70,
            "ROI5": 70,
            "ROI6": 70,
        },
        "lower_bound_target_percent": {
            "PTV": 95,
            "ROI1": 0,
            "ROI2": 0,
            "ROI3": 0,
            "ROI4": 0,
            "ROI5": 0,
            "ROI6": 0,
        },
        "higher_bound_target_percent": {
            "PTV": 100,
            "ROI1": 100,
            "ROI2": 100,
            "ROI3": 100,
            "ROI4": 75,
            "ROI5": 85,
            "ROI6": 96,
        },
        "weight": {
            "PTV": 10,
            "ROI1": 1,
            "ROI2": 2,
            "ROI3": 100,
            "ROI4": 2,
            "ROI5": 2,
            "ROI6": 1,
        },
    }

    constraints_w5 = {
        "lower_bound_gy": {
            "PTV": 74,
            "ROI1": 0,
            "ROI2": 0,
            "ROI3": 0,
            "ROI4": 0,
            "ROI5": 0,
            "ROI6": 0,
        },
        "higher_bound_gy": {
            "PTV": 83,
            "ROI1": 70,
            "ROI2": 50,
            "ROI3": 50,
            "ROI4": 70,
            "ROI5": 70,
            "ROI6": 70,
        },
        "lower_bound_target_percent": {
            "PTV": 95,
            "ROI1": 0,
            "ROI2": 0,
            "ROI3": 0,
            "ROI4": 0,
            "ROI5": 0,
            "ROI6": 0,
        },
        "higher_bound_target_percent": {
            "PTV": 100,
            "ROI1": 100,
            "ROI2": 100,
            "ROI3": 100,
            "ROI4": 75,
            "ROI5": 85,
            "ROI6": 96,
        },
        "weight": {
            "PTV": 10,
            "ROI1": 1,
            "ROI2": 2,
            "ROI3": 2,
            "ROI4": 100,
            "ROI5": 2,
            "ROI6": 1,
        },
    }

    constraints_w6 = {
        "lower_bound_gy": {
            "PTV": 74,
            "ROI1": 0,
            "ROI2": 0,
            "ROI3": 0,
            "ROI4": 0,
            "ROI5": 0,
            "ROI6": 0,
        },
        "higher_bound_gy": {
            "PTV": 83,
            "ROI1": 70,
            "ROI2": 50,
            "ROI3": 50,
            "ROI4": 70,
            "ROI5": 70,
            "ROI6": 70,
        },
        "lower_bound_target_percent": {
            "PTV": 95,
            "ROI1": 0,
            "ROI2": 0,
            "ROI3": 0,
            "ROI4": 0,
            "ROI5": 0,
            "ROI6": 0,
        },
        "higher_bound_target_percent": {
            "PTV": 100,
            "ROI1": 100,
            "ROI2": 100,
            "ROI3": 100,
            "ROI4": 75,
            "ROI5": 85,
            "ROI6": 96,
        },
        "weight": {
            "PTV": 10,
            "ROI1": 1,
            "ROI2": 2,
            "ROI3": 2,
            "ROI4": 2,
            "ROI5": 100,
            "ROI6": 1,
        },
    }

    constraints_w7 = {
        "lower_bound_gy": {
            "PTV": 74,
            "ROI1": 0,
            "ROI2": 0,
            "ROI3": 0,
            "ROI4": 0,
            "ROI5": 0,
            "ROI6": 0,
        },
        "higher_bound_gy": {
            "PTV": 83,
            "ROI1": 70,
            "ROI2": 50,
            "ROI3": 50,
            "ROI4": 70,
            "ROI5": 70,
            "ROI6": 70,
        },
        "lower_bound_target_percent": {
            "PTV": 95,
            "ROI1": 0,
            "ROI2": 0,
            "ROI3": 0,
            "ROI4": 0,
            "ROI5": 0,
            "ROI6": 0,
        },
        "higher_bound_target_percent": {
            "PTV": 100,
            "ROI1": 100,
            "ROI2": 100,
            "ROI3": 100,
            "ROI4": 75,
            "ROI5": 85,
            "ROI6": 96,
        },
        "weight": {
            "PTV": 10,
            "ROI1": 1,
            "ROI2": 2,
            "ROI3": 2,
            "ROI4": 2,
            "ROI5": 2,
            "ROI6": 100,
        },
    }

    list_constraints_weight = [
        constraints_w1,
        constraints_w2,
        constraints_w3,
        constraints_w4,
        constraints_w5,
        constraints_w6,
        constraints_w7,
    ]
