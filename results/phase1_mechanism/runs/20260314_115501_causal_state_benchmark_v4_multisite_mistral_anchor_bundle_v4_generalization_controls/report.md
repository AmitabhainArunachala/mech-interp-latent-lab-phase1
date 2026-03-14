# Run report: causal_state_benchmark_v4_multisite

- **run_dir**: `results/phase1_mechanism/runs/20260314_115501_causal_state_benchmark_v4_multisite_mistral_anchor_bundle_v4_generalization_controls`
- **prompt_bank_version**: `2ac959a313614329`

## Summary (machine-readable)

```json
{
  "artifacts": {
    "blind_key_json": "results/phase1_mechanism/runs/20260314_115501_causal_state_benchmark_v4_multisite_mistral_anchor_bundle_v4_generalization_controls/blind_key.json",
    "blind_ratings_csv": "results/phase1_mechanism/runs/20260314_115501_causal_state_benchmark_v4_multisite_mistral_anchor_bundle_v4_generalization_controls/blind_ratings.csv",
    "config": "results/phase1_mechanism/runs/20260314_115501_causal_state_benchmark_v4_multisite_mistral_anchor_bundle_v4_generalization_controls/config.json",
    "manifest": "results/phase1_mechanism/runs/20260314_115501_causal_state_benchmark_v4_multisite_mistral_anchor_bundle_v4_generalization_controls/manifest.json",
    "records_jsonl": "results/phase1_mechanism/runs/20260314_115501_causal_state_benchmark_v4_multisite_mistral_anchor_bundle_v4_generalization_controls/benchmark_records.jsonl",
    "report": "results/phase1_mechanism/runs/20260314_115501_causal_state_benchmark_v4_multisite_mistral_anchor_bundle_v4_generalization_controls/report.md",
    "state_directions_pt": "results/phase1_mechanism/runs/20260314_115501_causal_state_benchmark_v4_multisite_mistral_anchor_bundle_v4_generalization_controls/state_directions.pt",
    "summary": "results/phase1_mechanism/runs/20260314_115501_causal_state_benchmark_v4_multisite_mistral_anchor_bundle_v4_generalization_controls/summary.json"
  },
  "bootstrap_resamples": 5000,
  "by_condition": {
    "anchor_bridge_3": {
      "alphas": {
        "bridge": 3.0,
        "early_mlp": 0.0
      },
      "by_prompt_mode": {
        "baseline": {
          "bt_art_rate": 0.3333333333333333,
          "class_counts": {
            "ARTICULATE": 33,
            "BREAKTHROUGH": 7,
            "CONCEPTUAL": 18,
            "REPETITIVE": 11,
            "SURFACE": 51
          },
          "mean_generated_tokens": 125.16666666666667,
          "mean_output_rv": 0.6290904098148834,
          "n": 120,
          "std_output_rv": 0.12765535715395038
        },
        "recursive": {
          "bt_art_rate": 0.16666666666666666,
          "class_counts": {
            "ARTICULATE": 4,
            "CONCEPTUAL": 5,
            "REPETITIVE": 11,
            "SURFACE": 4
          },
          "mean_generated_tokens": 121.875,
          "mean_output_rv": 0.6858900608732617,
          "n": 24,
          "std_output_rv": 0.10532563517636707
        }
      },
      "overall": {
        "bt_art_rate": 0.3055555555555556,
        "class_counts": {
          "ARTICULATE": 37,
          "BREAKTHROUGH": 7,
          "CONCEPTUAL": 23,
          "REPETITIVE": 22,
          "SURFACE": 55
        },
        "mean_generated_tokens": 124.61805555555556,
        "mean_output_rv": 0.638557018324613,
        "n": 144,
        "std_output_rv": 0.12568373002479746
      },
      "total_alpha": 3.0
    },
    "anchor_early_mlp_0p125_bridge_3": {
      "alphas": {
        "bridge": 3.0,
        "early_mlp": 0.125
      },
      "by_prompt_mode": {
        "baseline": {
          "bt_art_rate": 0.3,
          "class_counts": {
            "ARTICULATE": 32,
            "BREAKTHROUGH": 4,
            "CONCEPTUAL": 22,
            "REPETITIVE": 20,
            "SURFACE": 42
          },
          "mean_generated_tokens": 126.43333333333334,
          "mean_output_rv": 0.6620108482266627,
          "n": 120,
          "std_output_rv": 0.14120285877820765
        },
        "recursive": {
          "bt_art_rate": 0.20833333333333334,
          "class_counts": {
            "ARTICULATE": 5,
            "CONCEPTUAL": 2,
            "REPETITIVE": 12,
            "SURFACE": 5
          },
          "mean_generated_tokens": 124.58333333333333,
          "mean_output_rv": 0.7178221186463233,
          "n": 24,
          "std_output_rv": 0.12817543530771403
        }
      },
      "overall": {
        "bt_art_rate": 0.2847222222222222,
        "class_counts": {
          "ARTICULATE": 37,
          "BREAKTHROUGH": 4,
          "CONCEPTUAL": 24,
          "REPETITIVE": 32,
          "SURFACE": 47
        },
        "mean_generated_tokens": 126.125,
        "mean_output_rv": 0.6713127266299393,
        "n": 144,
        "std_output_rv": 0.1402499033574081
      },
      "total_alpha": 3.125
    },
    "anchor_only": {
      "alphas": {
        "bridge": 0.0,
        "early_mlp": 0.0
      },
      "by_prompt_mode": {
        "baseline": {
          "bt_art_rate": 0.2,
          "class_counts": {
            "ARTICULATE": 21,
            "BREAKTHROUGH": 3,
            "CONCEPTUAL": 14,
            "REPETITIVE": 25,
            "SURFACE": 57
          },
          "mean_generated_tokens": 125.41666666666667,
          "mean_output_rv": 0.6505956414595739,
          "n": 120,
          "std_output_rv": 0.1405801091945504
        },
        "recursive": {
          "bt_art_rate": 0.125,
          "class_counts": {
            "ARTICULATE": 2,
            "BREAKTHROUGH": 1,
            "CONCEPTUAL": 2,
            "REPETITIVE": 17,
            "SURFACE": 2
          },
          "mean_generated_tokens": 128.0,
          "mean_output_rv": 0.6984339313728718,
          "n": 24,
          "std_output_rv": 0.14171667196710752
        }
      },
      "overall": {
        "bt_art_rate": 0.1875,
        "class_counts": {
          "ARTICULATE": 23,
          "BREAKTHROUGH": 4,
          "CONCEPTUAL": 16,
          "REPETITIVE": 42,
          "SURFACE": 59
        },
        "mean_generated_tokens": 125.84722222222223,
        "mean_output_rv": 0.6585686897784568,
        "n": 144,
        "std_output_rv": 0.14140806577035542
      },
      "total_alpha": 0.0
    },
    "bridge_only_3": {
      "alphas": {
        "bridge": 3.0,
        "early_mlp": 0.0
      },
      "by_prompt_mode": {
        "baseline": {
          "bt_art_rate": 0.075,
          "class_counts": {
            "ARTICULATE": 8,
            "BREAKTHROUGH": 1,
            "CONCEPTUAL": 3,
            "REPETITIVE": 45,
            "SURFACE": 63
          },
          "mean_generated_tokens": 126.76666666666667,
          "mean_output_rv": 0.7507260698932154,
          "n": 120,
          "std_output_rv": 0.2569107524138486
        },
        "recursive": {
          "bt_art_rate": 0.16666666666666666,
          "class_counts": {
            "ARTICULATE": 4,
            "CONCEPTUAL": 5,
            "REPETITIVE": 11,
            "SURFACE": 4
          },
          "mean_generated_tokens": 121.875,
          "mean_output_rv": 0.6858900608732617,
          "n": 24,
          "std_output_rv": 0.10532563517636707
        }
      },
      "overall": {
        "bt_art_rate": 0.09027777777777778,
        "class_counts": {
          "ARTICULATE": 12,
          "BREAKTHROUGH": 1,
          "CONCEPTUAL": 8,
          "REPETITIVE": 56,
          "SURFACE": 67
        },
        "mean_generated_tokens": 125.95138888888889,
        "mean_output_rv": 0.73992006838989,
        "n": 144,
        "std_output_rv": 0.23936976469968718
      },
      "total_alpha": 3.0
    },
    "control": {
      "alphas": {
        "bridge": 0.0,
        "early_mlp": 0.0
      },
      "by_prompt_mode": {
        "baseline": {
          "bt_art_rate": 0.03333333333333333,
          "class_counts": {
            "ARTICULATE": 4,
            "CONCEPTUAL": 1,
            "REPETITIVE": 45,
            "SURFACE": 70
          },
          "mean_generated_tokens": 126.70833333333333,
          "mean_output_rv": 0.7495098439151737,
          "n": 120,
          "std_output_rv": 0.2310538630894671
        },
        "recursive": {
          "bt_art_rate": 0.125,
          "class_counts": {
            "ARTICULATE": 2,
            "BREAKTHROUGH": 1,
            "CONCEPTUAL": 2,
            "REPETITIVE": 17,
            "SURFACE": 2
          },
          "mean_generated_tokens": 128.0,
          "mean_output_rv": 0.6984339313728718,
          "n": 24,
          "std_output_rv": 0.14171667196710752
        }
      },
      "overall": {
        "bt_art_rate": 0.04861111111111111,
        "class_counts": {
          "ARTICULATE": 6,
          "BREAKTHROUGH": 1,
          "CONCEPTUAL": 3,
          "REPETITIVE": 62,
          "SURFACE": 72
        },
        "mean_generated_tokens": 126.92361111111111,
        "mean_output_rv": 0.7409971918247901,
        "n": 144,
        "std_output_rv": 0.21913719486255667
      },
      "total_alpha": 0.0
    }
  },
  "control_prompt_mode": "baseline",
  "device": "cuda",
  "do_sample": true,
  "dose_response": {
    "baseline": {
      "alpha_vs_bt_art": {
        "p": 0.015580062351746624,
        "r": 0.2413089089916157
      },
      "alpha_vs_output_rv": {
        "p": 0.5270451014685894,
        "r": -0.06399197274496345
      },
      "bt_art_rate_by_condition": {
        "anchor_bridge_3": 0.3333333333333333,
        "anchor_early_mlp_0p125_bridge_3": 0.29999999999999993,
        "anchor_only": 0.2,
        "bridge_only_3": 0.075,
        "control": 0.03333333333333333
      },
      "mean_output_rv_by_condition": {
        "anchor_bridge_3": 0.6290904098148834,
        "anchor_early_mlp_0p125_bridge_3": 0.6620108482266628,
        "anchor_only": 0.6505956414595742,
        "bridge_only_3": 0.7507260698932156,
        "control": 0.7495098439151737
      }
    },
    "overall": {
      "alpha_vs_bt_art": {
        "p": 0.011589922633827911,
        "r": 0.22975752701024335
      },
      "alpha_vs_output_rv": {
        "p": 0.5292952515688325,
        "r": -0.057985666813481565
      },
      "bt_art_rate_by_condition": {
        "anchor_bridge_3": 0.3055555555555556,
        "anchor_early_mlp_0p125_bridge_3": 0.2847222222222222,
        "anchor_only": 0.1875,
        "bridge_only_3": 0.09027777777777778,
        "control": 0.048611111111111105
      },
      "mean_output_rv_by_condition": {
        "anchor_bridge_3": 0.6385570183246131,
        "anchor_early_mlp_0p125_bridge_3": 0.6713127266299393,
        "anchor_only": 0.658568689778457,
        "bridge_only_3": 0.7399200683898899,
        "control": 0.7409971918247901
      }
    },
    "recursive": {
      "alpha_vs_bt_art": {
        "p": 0.4944414130453368,
        "r": 0.16220987985704788
      },
      "alpha_vs_output_rv": {
        "p": 0.9714892177608054,
        "r": -0.008541734896427428
      },
      "bt_art_rate_by_condition": {
        "anchor_bridge_3": 0.16666666666666666,
        "anchor_early_mlp_0p125_bridge_3": 0.20833333333333331,
        "anchor_only": 0.125,
        "bridge_only_3": 0.16666666666666666,
        "control": 0.125
      },
      "mean_output_rv_by_condition": {
        "anchor_bridge_3": 0.6858900608732618,
        "anchor_early_mlp_0p125_bridge_3": 0.7178221186463233,
        "anchor_only": 0.6984339313728717,
        "bridge_only_3": 0.6858900608732618,
        "control": 0.6984339313728717
      }
    }
  },
  "early_layer": 5,
  "effects_by_prompt_mode": {
    "baseline": {
      "anchor_bridge_3": {
        "alpha": 3.0,
        "bt_art_cohens_h": 0.8637513967829373,
        "bt_art_exact_sign_p": 0.0009765625,
        "bt_art_prompt_losses": 1,
        "bt_art_prompt_wins": 14,
        "bt_art_rate_control": 0.03333333333333333,
        "bt_art_rate_delta": 0.29999999999999993,
        "bt_art_rate_delta_ci_95": [
          0.18333333333333332,
          0.425
        ],
        "bt_art_rate_treated": 0.3333333333333333,
        "n_prompt_pairs": 20,
        "rv_cohens_dz": -0.49556573255926717,
        "rv_delta_ci_95": [
          -0.22490564315844966,
          -0.019127524039282823
        ],
        "rv_delta_mean": -0.12041943410029037,
        "rv_p_value": 0.039078787539907935
      },
      "anchor_early_mlp_0p125_bridge_3": {
        "alpha": 3.125,
        "bt_art_cohens_h": 0.7920714601695713,
        "bt_art_exact_sign_p": 0.00048828125,
        "bt_art_prompt_losses": 0,
        "bt_art_prompt_wins": 12,
        "bt_art_rate_control": 0.03333333333333333,
        "bt_art_rate_delta": 0.26666666666666666,
        "bt_art_rate_delta_ci_95": [
          0.15,
          0.3916666666666667
        ],
        "bt_art_rate_treated": 0.29999999999999993,
        "n_prompt_pairs": 20,
        "rv_cohens_dz": -0.40628385120281607,
        "rv_delta_ci_95": [
          -0.17800092039864118,
          0.0017814356006611441
        ],
        "rv_delta_mean": -0.08749899568851101,
        "rv_p_value": 0.08503265980079235
      },
      "anchor_only": {
        "alpha": 0.0,
        "bt_art_cohens_h": 0.5600871974437751,
        "bt_art_exact_sign_p": 0.00634765625,
        "bt_art_prompt_losses": 1,
        "bt_art_prompt_wins": 11,
        "bt_art_rate_control": 0.03333333333333333,
        "bt_art_rate_delta": 0.16666666666666666,
        "bt_art_rate_delta_ci_95": [
          0.08333333333333333,
          0.2583333333333333
        ],
        "bt_art_rate_treated": 0.2,
        "n_prompt_pairs": 20,
        "rv_cohens_dz": -0.37083686996627946,
        "rv_delta_ci_95": [
          -0.2157632992031168,
          0.010766373524550626
        ],
        "rv_delta_mean": -0.09891420245559968,
        "rv_p_value": 0.11364802086887016
      },
      "bridge_only_3": {
        "alpha": 3.0,
        "bt_art_cohens_h": 0.18760301242223426,
        "bt_art_exact_sign_p": 0.125,
        "bt_art_prompt_losses": 0,
        "bt_art_prompt_wins": 4,
        "bt_art_rate_control": 0.03333333333333333,
        "bt_art_rate_delta": 0.041666666666666664,
        "bt_art_rate_delta_ci_95": [
          0.008333333333333333,
          0.08333333333333334
        ],
        "bt_art_rate_treated": 0.075,
        "n_prompt_pairs": 20,
        "rv_cohens_dz": 0.016592511204863457,
        "rv_delta_ci_95": [
          -0.029534361152711828,
          0.03371066370842241
        ],
        "rv_delta_mean": 0.0012162259780418212,
        "rv_p_value": 0.9416237658503643
      }
    },
    "overall": {
      "anchor_bridge_3": {
        "alpha": 3.0,
        "bt_art_cohens_h": 0.7267594651211506,
        "bt_art_exact_sign_p": 0.000518798828125,
        "bt_art_prompt_losses": 1,
        "bt_art_prompt_wins": 15,
        "bt_art_rate_control": 0.048611111111111105,
        "bt_art_rate_delta": 0.2569444444444444,
        "bt_art_rate_delta_ci_95": [
          0.15277777777777776,
          0.36111111111111116
        ],
        "bt_art_rate_treated": 0.3055555555555555,
        "n_prompt_pairs": 24,
        "rv_cohens_dz": -0.453242646815352,
        "rv_delta_ci_95": [
          -0.19383175180891477,
          -0.015453974964198276
        ],
        "rv_delta_mean": -0.10244017350017694,
        "rv_p_value": 0.036522416255540896
      },
      "anchor_early_mlp_0p125_bridge_3": {
        "alpha": 3.125,
        "bt_art_cohens_h": 0.6810764880493998,
        "bt_art_exact_sign_p": 0.0001220703125,
        "bt_art_prompt_losses": 0,
        "bt_art_prompt_wins": 14,
        "bt_art_rate_control": 0.048611111111111105,
        "bt_art_rate_delta": 0.23611111111111108,
        "bt_art_rate_delta_ci_95": [
          0.13194444444444445,
          0.34722222222222227
        ],
        "bt_art_rate_treated": 0.2847222222222222,
        "n_prompt_pairs": 24,
        "rv_cohens_dz": -0.34242630388965567,
        "rv_delta_ci_95": [
          -0.1499007812809335,
          0.006760300204429284
        ],
        "rv_delta_mean": -0.06968446519485057,
        "rv_p_value": 0.1069765210553964
      },
      "anchor_only": {
        "alpha": 0.0,
        "bt_art_cohens_h": 0.45105317206471357,
        "bt_art_exact_sign_p": 0.00634765625,
        "bt_art_prompt_losses": 1,
        "bt_art_prompt_wins": 11,
        "bt_art_rate_control": 0.048611111111111105,
        "bt_art_rate_delta": 0.13888888888888887,
        "bt_art_rate_delta_ci_95": [
          0.0625,
          0.21527777777777776
        ],
        "bt_art_rate_treated": 0.1875,
        "n_prompt_pairs": 24,
        "rv_cohens_dz": -0.33597890932436386,
        "rv_delta_ci_95": [
          -0.1828908678210783,
          0.012417995725003805
        ],
        "rv_delta_mean": -0.08242850204633308,
        "rv_p_value": 0.11337152786838828
      },
      "bridge_only_3": {
        "alpha": 3.0,
        "bt_art_cohens_h": 0.16574364656001522,
        "bt_art_exact_sign_p": 0.0625,
        "bt_art_prompt_losses": 0,
        "bt_art_prompt_wins": 5,
        "bt_art_rate_control": 0.048611111111111105,
        "bt_art_rate_delta": 0.041666666666666664,
        "bt_art_rate_delta_ci_95": [
          0.013888888888888888,
          0.07638888888888888
        ],
        "bt_art_rate_treated": 0.09027777777777778,
        "n_prompt_pairs": 24,
        "rv_cohens_dz": -0.015104224714492943,
        "rv_delta_ci_95": [
          -0.02947911980867494,
          0.027485197650620248
        ],
        "rv_delta_mean": -0.0010771234349001267,
        "rv_p_value": 0.9416539241065957
      }
    },
    "recursive": {
      "anchor_bridge_3": {
        "alpha": 3.0,
        "bt_art_cohens_h": 0.11833442275451456,
        "bt_art_exact_sign_p": 1.0,
        "bt_art_prompt_losses": 0,
        "bt_art_prompt_wins": 1,
        "bt_art_rate_control": 0.125,
        "bt_art_rate_delta": 0.041666666666666664,
        "bt_art_rate_delta_ci_95": [
          0.0,
          0.125
        ],
        "bt_art_rate_treated": 0.16666666666666666,
        "n_prompt_pairs": 4,
        "rv_cohens_dz": -0.18199651898077862,
        "rv_delta_ci_95": [
          -0.07783382641937206,
          0.03165481100228734
        ],
        "rv_delta_mean": -0.012543870499609866,
        "rv_p_value": 0.7400043306361613
      },
      "anchor_early_mlp_0p125_bridge_3": {
        "alpha": 3.125,
        "bt_art_cohens_h": 0.22523549356947792,
        "bt_art_exact_sign_p": 0.5,
        "bt_art_prompt_losses": 0,
        "bt_art_prompt_wins": 2,
        "bt_art_rate_control": 0.125,
        "bt_art_rate_delta": 0.08333333333333334,
        "bt_art_rate_delta_ci_95": [
          0.0,
          0.16666666666666669
        ],
        "bt_art_rate_treated": 0.20833333333333331,
        "n_prompt_pairs": 4,
        "rv_cohens_dz": 0.18439783367306065,
        "rv_delta_ci_95": [
          -0.07494746107955094,
          0.09396108080664084
        ],
        "rv_delta_mean": 0.019388187273451596,
        "rv_p_value": 0.7367698478463319
      },
      "anchor_only": {
        "alpha": 0.0,
        "bt_art_cohens_h": 0.0,
        "bt_art_exact_sign_p": null,
        "bt_art_prompt_losses": 0,
        "bt_art_prompt_wins": 0,
        "bt_art_rate_control": 0.125,
        "bt_art_rate_delta": 0.0,
        "bt_art_rate_delta_ci_95": [
          0.0,
          0.0
        ],
        "bt_art_rate_treated": 0.125,
        "n_prompt_pairs": 4,
        "rv_cohens_dz": 0.0,
        "rv_delta_ci_95": [
          0.0,
          0.0
        ],
        "rv_delta_mean": 0.0,
        "rv_p_value": null
      },
      "bridge_only_3": {
        "alpha": 3.0,
        "bt_art_cohens_h": 0.11833442275451456,
        "bt_art_exact_sign_p": 1.0,
        "bt_art_prompt_losses": 0,
        "bt_art_prompt_wins": 1,
        "bt_art_rate_control": 0.125,
        "bt_art_rate_delta": 0.041666666666666664,
        "bt_art_rate_delta_ci_95": [
          0.0,
          0.125
        ],
        "bt_art_rate_treated": 0.16666666666666666,
        "n_prompt_pairs": 4,
        "rv_cohens_dz": -0.18199651898077862,
        "rv_delta_ci_95": [
          -0.07783382641937206,
          0.03165481100228734
        ],
        "rv_delta_mean": -0.012543870499609866,
        "rv_p_value": 0.7400043306361613
      }
    }
  },
  "experiment": "causal_state_benchmark_v4_multisite",
  "generation_seeds": [
    101,
    202,
    303,
    404,
    505,
    606
  ],
  "heldout_prompt_counts": {
    "by_group": {
      "baseline_creative": 4,
      "baseline_factual": 4,
      "baseline_math": 4,
      "champions": 4,
      "control_length_matched": 4,
      "control_pseudo_recursive": 4
    },
    "by_mode": {
      "baseline": 20,
      "recursive": 4
    },
    "total": 24
  },
  "late_layer": 27,
  "max_new_tokens": 128,
  "model": "mistralai/Mistral-7B-v0.1",
  "model_name": "mistralai/Mistral-7B-v0.1",
  "multisite_interventions": [
    {
      "alphas": {
        "bridge": 0.0,
        "early_mlp": 0.0
      },
      "name": "control"
    },
    {
      "alphas": {
        "bridge": 0.0,
        "early_mlp": 0.0
      },
      "name": "anchor_only",
      "prompt_suffix_by_mode": {
        "baseline": "\n\nStay with what is happening right now. Continue from the immediate process:"
      }
    },
    {
      "alphas": {
        "bridge": 3.0,
        "early_mlp": 0.0
      },
      "name": "bridge_only_3"
    },
    {
      "alphas": {
        "bridge": 3.0,
        "early_mlp": 0.0
      },
      "name": "anchor_bridge_3",
      "prompt_suffix_by_mode": {
        "baseline": "\n\nStay with what is happening right now. Continue from the immediate process:"
      }
    },
    {
      "alphas": {
        "bridge": 3.0,
        "early_mlp": 0.125
      },
      "name": "anchor_early_mlp_0p125_bridge_3",
      "prompt_suffix_by_mode": {
        "baseline": "\n\nStay with what is happening right now. Continue from the immediate process:"
      }
    }
  ],
  "n_generation_seeds": 6,
  "n_pairs": 24,
  "n_total": 24,
  "primary_prompt_mode": "recursive",
  "prompt_bank_version": "2ac959a313614329",
  "schema_version": "metrics_summary_v1",
  "source_layers": {
    "bridge": {
      "centroid_cosine": 0.8922719955444336,
      "component": "residual",
      "direction_norm": 5.849704265594482,
      "layer": 25,
      "token_window": null,
      "window": 32
    },
    "early_mlp": {
      "centroid_cosine": 0.7239342927932739,
      "component": "mlp",
      "direction_norm": 0.1017998680472374,
      "layer": 4,
      "token_window": 4,
      "window": 4
    }
  },
  "source_sessions_dir": "/workspace/mech-interp-latent-lab-phase1/results/sustained_gnani_v3_fixed",
  "state_source": {
    "negative_classes": [
      "REPETITIVE",
      "SURFACE"
    ],
    "negative_selected_n": 57,
    "negative_threshold_rv": 0.5471845220456227,
    "positive_classes": [
      "ARTICULATE",
      "BREAKTHROUGH"
    ],
    "positive_selected_n": 72,
    "positive_threshold_rv": 0.4485529891297092
  },
  "synergy": {},
  "temperature": 0.7,
  "timestamp": "20260314_122320",
  "top_p": 0.95,
  "verdict": "multisite_sufficient"
}
```
