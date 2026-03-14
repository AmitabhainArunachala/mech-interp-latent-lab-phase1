# Run report: causal_state_benchmark_v4_multisite

- **run_dir**: `results/phase1_mechanism/runs/20260314_133516_causal_state_benchmark_v4_multisite_mistral_anchor_bundle_v5_ordinary_baselines_confirmatory`
- **prompt_bank_version**: `2ac959a313614329`

## Summary (machine-readable)

```json
{
  "artifacts": {
    "blind_key_json": "results/phase1_mechanism/runs/20260314_133516_causal_state_benchmark_v4_multisite_mistral_anchor_bundle_v5_ordinary_baselines_confirmatory/blind_key.json",
    "blind_ratings_csv": "results/phase1_mechanism/runs/20260314_133516_causal_state_benchmark_v4_multisite_mistral_anchor_bundle_v5_ordinary_baselines_confirmatory/blind_ratings.csv",
    "config": "results/phase1_mechanism/runs/20260314_133516_causal_state_benchmark_v4_multisite_mistral_anchor_bundle_v5_ordinary_baselines_confirmatory/config.json",
    "manifest": "results/phase1_mechanism/runs/20260314_133516_causal_state_benchmark_v4_multisite_mistral_anchor_bundle_v5_ordinary_baselines_confirmatory/manifest.json",
    "records_jsonl": "results/phase1_mechanism/runs/20260314_133516_causal_state_benchmark_v4_multisite_mistral_anchor_bundle_v5_ordinary_baselines_confirmatory/benchmark_records.jsonl",
    "report": "results/phase1_mechanism/runs/20260314_133516_causal_state_benchmark_v4_multisite_mistral_anchor_bundle_v5_ordinary_baselines_confirmatory/report.md",
    "state_directions_pt": "results/phase1_mechanism/runs/20260314_133516_causal_state_benchmark_v4_multisite_mistral_anchor_bundle_v5_ordinary_baselines_confirmatory/state_directions.pt",
    "summary": "results/phase1_mechanism/runs/20260314_133516_causal_state_benchmark_v4_multisite_mistral_anchor_bundle_v5_ordinary_baselines_confirmatory/summary.json"
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
          "bt_art_rate": 0.16666666666666666,
          "class_counts": {
            "ARTICULATE": 14,
            "BREAKTHROUGH": 2,
            "CONCEPTUAL": 16,
            "REPETITIVE": 15,
            "SURFACE": 49
          },
          "mean_generated_tokens": 126.27083333333333,
          "mean_output_rv": 0.6464459039005196,
          "n": 96,
          "std_output_rv": 0.1399965611633329
        },
        "recursive": {
          "bt_art_rate": 0.1875,
          "class_counts": {
            "ARTICULATE": 6,
            "CONCEPTUAL": 5,
            "REPETITIVE": 16,
            "SURFACE": 5
          },
          "mean_generated_tokens": 122.9375,
          "mean_output_rv": 0.6800018699557366,
          "n": 32,
          "std_output_rv": 0.11797781368235842
        }
      },
      "overall": {
        "bt_art_rate": 0.171875,
        "class_counts": {
          "ARTICULATE": 20,
          "BREAKTHROUGH": 2,
          "CONCEPTUAL": 21,
          "REPETITIVE": 31,
          "SURFACE": 54
        },
        "mean_generated_tokens": 125.4375,
        "mean_output_rv": 0.6548348954143238,
        "n": 128,
        "std_output_rv": 0.135170184268183
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
          "bt_art_rate": 0.15625,
          "class_counts": {
            "ARTICULATE": 13,
            "BREAKTHROUGH": 2,
            "CONCEPTUAL": 22,
            "REPETITIVE": 12,
            "SURFACE": 47
          },
          "mean_generated_tokens": 127.07291666666667,
          "mean_output_rv": 0.6733656154821692,
          "n": 96,
          "std_output_rv": 0.14274542542720908
        },
        "recursive": {
          "bt_art_rate": 0.21875,
          "class_counts": {
            "ARTICULATE": 7,
            "CONCEPTUAL": 2,
            "REPETITIVE": 18,
            "SURFACE": 5
          },
          "mean_generated_tokens": 125.4375,
          "mean_output_rv": 0.7137864404098788,
          "n": 32,
          "std_output_rv": 0.1265030303206266
        }
      },
      "overall": {
        "bt_art_rate": 0.171875,
        "class_counts": {
          "ARTICULATE": 20,
          "BREAKTHROUGH": 2,
          "CONCEPTUAL": 24,
          "REPETITIVE": 30,
          "SURFACE": 52
        },
        "mean_generated_tokens": 126.6640625,
        "mean_output_rv": 0.6834708217140966,
        "n": 128,
        "std_output_rv": 0.1394886645363463
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
          "bt_art_rate": 0.0625,
          "class_counts": {
            "ARTICULATE": 6,
            "CONCEPTUAL": 8,
            "REPETITIVE": 15,
            "SURFACE": 67
          },
          "mean_generated_tokens": 125.58333333333333,
          "mean_output_rv": 0.6908794455987851,
          "n": 96,
          "std_output_rv": 0.14633978506252474
        },
        "recursive": {
          "bt_art_rate": 0.09375,
          "class_counts": {
            "ARTICULATE": 2,
            "BREAKTHROUGH": 1,
            "CONCEPTUAL": 2,
            "REPETITIVE": 24,
            "SURFACE": 3
          },
          "mean_generated_tokens": 128.0,
          "mean_output_rv": 0.7162827964497804,
          "n": 32,
          "std_output_rv": 0.13609461572226383
        }
      },
      "overall": {
        "bt_art_rate": 0.0703125,
        "class_counts": {
          "ARTICULATE": 8,
          "BREAKTHROUGH": 1,
          "CONCEPTUAL": 10,
          "REPETITIVE": 39,
          "SURFACE": 70
        },
        "mean_generated_tokens": 126.1875,
        "mean_output_rv": 0.6972302833115339,
        "n": 128,
        "std_output_rv": 0.1437440477005845
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
          "bt_art_rate": 0.0625,
          "class_counts": {
            "ARTICULATE": 6,
            "CONCEPTUAL": 4,
            "REPETITIVE": 5,
            "SURFACE": 81
          },
          "mean_generated_tokens": 124.69791666666667,
          "mean_output_rv": 0.6166278053210595,
          "n": 96,
          "std_output_rv": 0.19924951845400135
        },
        "recursive": {
          "bt_art_rate": 0.1875,
          "class_counts": {
            "ARTICULATE": 6,
            "CONCEPTUAL": 5,
            "REPETITIVE": 16,
            "SURFACE": 5
          },
          "mean_generated_tokens": 122.9375,
          "mean_output_rv": 0.6800018699557366,
          "n": 32,
          "std_output_rv": 0.11797781368235842
        }
      },
      "overall": {
        "bt_art_rate": 0.09375,
        "class_counts": {
          "ARTICULATE": 12,
          "CONCEPTUAL": 9,
          "REPETITIVE": 21,
          "SURFACE": 86
        },
        "mean_generated_tokens": 124.2578125,
        "mean_output_rv": 0.6324713214797288,
        "n": 128,
        "std_output_rv": 0.1839934832206421
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
          "bt_art_rate": 0.03125,
          "class_counts": {
            "ARTICULATE": 3,
            "CONCEPTUAL": 1,
            "REPETITIVE": 5,
            "SURFACE": 87
          },
          "mean_generated_tokens": 124.8125,
          "mean_output_rv": 0.6128053275031469,
          "n": 96,
          "std_output_rv": 0.15925641053551604
        },
        "recursive": {
          "bt_art_rate": 0.09375,
          "class_counts": {
            "ARTICULATE": 2,
            "BREAKTHROUGH": 1,
            "CONCEPTUAL": 2,
            "REPETITIVE": 24,
            "SURFACE": 3
          },
          "mean_generated_tokens": 128.0,
          "mean_output_rv": 0.7162827964497804,
          "n": 32,
          "std_output_rv": 0.13609461572226383
        }
      },
      "overall": {
        "bt_art_rate": 0.046875,
        "class_counts": {
          "ARTICULATE": 5,
          "BREAKTHROUGH": 1,
          "CONCEPTUAL": 3,
          "REPETITIVE": 29,
          "SURFACE": 90
        },
        "mean_generated_tokens": 125.609375,
        "mean_output_rv": 0.6386746947398052,
        "n": 128,
        "std_output_rv": 0.15973904811026754
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
        "p": 0.00487369574036073,
        "r": 0.35880938892835196
      },
      "alpha_vs_output_rv": {
        "p": 0.8171130142959439,
        "r": -0.030489956239751387
      },
      "bt_art_rate_by_condition": {
        "anchor_bridge_3": 0.16666666666666666,
        "anchor_early_mlp_0p125_bridge_3": 0.15625,
        "anchor_only": 0.0625,
        "bridge_only_3": 0.0625,
        "control": 0.03125
      },
      "mean_output_rv_by_condition": {
        "anchor_bridge_3": 0.6464459039005195,
        "anchor_early_mlp_0p125_bridge_3": 0.6733656154821693,
        "anchor_only": 0.6908794455987852,
        "bridge_only_3": 0.6166278053210597,
        "control": 0.6128053275031468
      }
    },
    "overall": {
      "alpha_vs_bt_art": {
        "p": 0.001709028220114225,
        "r": 0.3452682693814968
      },
      "alpha_vs_output_rv": {
        "p": 0.6093239666140711,
        "r": -0.058000569038435756
      },
      "bt_art_rate_by_condition": {
        "anchor_bridge_3": 0.171875,
        "anchor_early_mlp_0p125_bridge_3": 0.171875,
        "anchor_only": 0.0703125,
        "bridge_only_3": 0.09375,
        "control": 0.046875
      },
      "mean_output_rv_by_condition": {
        "anchor_bridge_3": 0.6548348954143238,
        "anchor_early_mlp_0p125_bridge_3": 0.6834708217140966,
        "anchor_only": 0.697230283311534,
        "bridge_only_3": 0.6324713214797288,
        "control": 0.6386746947398052
      }
    },
    "recursive": {
      "alpha_vs_bt_art": {
        "p": 0.13175588794438317,
        "r": 0.34879376450995686
      },
      "alpha_vs_output_rv": {
        "p": 0.4518676768078774,
        "r": -0.17834854946808873
      },
      "bt_art_rate_by_condition": {
        "anchor_bridge_3": 0.1875,
        "anchor_early_mlp_0p125_bridge_3": 0.21875,
        "anchor_only": 0.09375,
        "bridge_only_3": 0.1875,
        "control": 0.09375
      },
      "mean_output_rv_by_condition": {
        "anchor_bridge_3": 0.6800018699557367,
        "anchor_early_mlp_0p125_bridge_3": 0.7137864404098788,
        "anchor_only": 0.7162827964497803,
        "bridge_only_3": 0.6800018699557367,
        "control": 0.7162827964497803
      }
    }
  },
  "early_layer": 5,
  "effects_by_prompt_mode": {
    "baseline": {
      "anchor_bridge_3": {
        "alpha": 3.0,
        "bt_art_cohens_h": 0.4856474688777067,
        "bt_art_exact_sign_p": 0.0703125,
        "bt_art_prompt_losses": 1,
        "bt_art_prompt_wins": 7,
        "bt_art_rate_control": 0.03125,
        "bt_art_rate_delta": 0.13541666666666666,
        "bt_art_rate_delta_ci_95": [
          0.041666666666666664,
          0.23958333333333334
        ],
        "bt_art_rate_treated": 0.16666666666666666,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": 0.28720805045811804,
        "rv_delta_ci_95": [
          -0.02878838165522642,
          0.09660217512913012
        ],
        "rv_delta_mean": 0.0336405763973727,
        "rv_p_value": 0.34115845813008217
      },
      "anchor_early_mlp_0p125_bridge_3": {
        "alpha": 3.125,
        "bt_art_cohens_h": 0.45733435967843716,
        "bt_art_exact_sign_p": 0.00390625,
        "bt_art_prompt_losses": 0,
        "bt_art_prompt_wins": 9,
        "bt_art_rate_control": 0.03125,
        "bt_art_rate_delta": 0.125,
        "bt_art_rate_delta_ci_95": [
          0.07291666666666667,
          0.1875
        ],
        "bt_art_rate_treated": 0.15625,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": 0.7323956469161095,
        "rv_delta_ci_95": [
          0.016035734472121962,
          0.10373646374688536
        ],
        "rv_delta_mean": 0.06056028797902239,
        "rv_p_value": 0.027622530814565397
      },
      "anchor_only": {
        "alpha": 0.0,
        "bt_art_cohens_h": 0.14993930859393378,
        "bt_art_exact_sign_p": 0.625,
        "bt_art_prompt_losses": 1,
        "bt_art_prompt_wins": 3,
        "bt_art_rate_control": 0.03125,
        "bt_art_rate_delta": 0.03125,
        "bt_art_rate_delta_ci_95": [
          -0.020833333333333332,
          0.08333333333333333
        ],
        "bt_art_rate_treated": 0.0625,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": 0.7708364867273935,
        "rv_delta_ci_95": [
          0.022139770019977904,
          0.1324364879456704
        ],
        "rv_delta_mean": 0.07807411809563837,
        "rv_p_value": 0.02178404823327559
      },
      "bridge_only_3": {
        "alpha": 3.0,
        "bt_art_cohens_h": 0.14993930859393378,
        "bt_art_exact_sign_p": 0.25,
        "bt_art_prompt_losses": 0,
        "bt_art_prompt_wins": 3,
        "bt_art_rate_control": 0.03125,
        "bt_art_rate_delta": 0.03125,
        "bt_art_rate_delta_ci_95": [
          0.0,
          0.0625
        ],
        "bt_art_rate_treated": 0.0625,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": 0.0591939397410239,
        "rv_delta_ci_95": [
          -0.030737703412364828,
          0.03902441068062801
        ],
        "rv_delta_mean": 0.003822477817912784,
        "rv_p_value": 0.8412751872465943
      }
    },
    "overall": {
      "anchor_bridge_3": {
        "alpha": 3.0,
        "bt_art_cohens_h": 0.41848923783486613,
        "bt_art_exact_sign_p": 0.01171875,
        "bt_art_prompt_losses": 1,
        "bt_art_prompt_wins": 10,
        "bt_art_rate_control": 0.046875,
        "bt_art_rate_delta": 0.125,
        "bt_art_rate_delta_ci_95": [
          0.0546875,
          0.203125
        ],
        "bt_art_rate_treated": 0.171875,
        "n_prompt_pairs": 16,
        "rv_cohens_dz": 0.1508879344616175,
        "rv_delta_ci_95": [
          -0.034579288859974466,
          0.06801349763575429
        ],
        "rv_delta_mean": 0.016160200674518607,
        "rv_p_value": 0.555159585622967
      },
      "anchor_early_mlp_0p125_bridge_3": {
        "alpha": 3.125,
        "bt_art_cohens_h": 0.41848923783486613,
        "bt_art_exact_sign_p": 0.00048828125,
        "bt_art_prompt_losses": 0,
        "bt_art_prompt_wins": 12,
        "bt_art_rate_control": 0.046875,
        "bt_art_rate_delta": 0.125,
        "bt_art_rate_delta_ci_95": [
          0.078125,
          0.171875
        ],
        "bt_art_rate_treated": 0.171875,
        "n_prompt_pairs": 16,
        "rv_cohens_dz": 0.5298318339419491,
        "rv_delta_ci_95": [
          0.0033936044199706825,
          0.08347148846175956
        ],
        "rv_delta_mean": 0.0447961269742914,
        "rv_p_value": 0.05115790329568865
      },
      "anchor_only": {
        "alpha": 0.0,
        "bt_art_cohens_h": 0.100281148493792,
        "bt_art_exact_sign_p": 0.625,
        "bt_art_prompt_losses": 1,
        "bt_art_prompt_wins": 3,
        "bt_art_rate_control": 0.046875,
        "bt_art_rate_delta": 0.0234375,
        "bt_art_rate_delta_ci_95": [
          -0.015625,
          0.0625
        ],
        "bt_art_rate_treated": 0.0703125,
        "n_prompt_pairs": 16,
        "rv_cohens_dz": 0.6262678581140839,
        "rv_delta_ci_95": [
          0.01636938636768528,
          0.10358105972143322
        ],
        "rv_delta_mean": 0.05855558857172878,
        "rv_p_value": 0.024261671344849127
      },
      "bridge_only_3": {
        "alpha": 3.0,
        "bt_art_cohens_h": 0.18589945982010136,
        "bt_art_exact_sign_p": 0.03125,
        "bt_art_prompt_losses": 0,
        "bt_art_prompt_wins": 6,
        "bt_art_rate_control": 0.046875,
        "bt_art_rate_delta": 0.046875,
        "bt_art_rate_delta_ci_95": [
          0.015625,
          0.078125
        ],
        "bt_art_rate_treated": 0.09375,
        "n_prompt_pairs": 16,
        "rv_cohens_dz": -0.10047946773895378,
        "rv_delta_ci_95": [
          -0.03485240868460744,
          0.022294075667774152
        ],
        "rv_delta_mean": -0.006203373260076334,
        "rv_p_value": 0.693413002990824
      }
    },
    "recursive": {
      "anchor_bridge_3": {
        "alpha": 3.0,
        "bt_art_cohens_h": 0.2732963053028443,
        "bt_art_exact_sign_p": 0.25,
        "bt_art_prompt_losses": 0,
        "bt_art_prompt_wins": 3,
        "bt_art_rate_control": 0.09375,
        "bt_art_rate_delta": 0.09375,
        "bt_art_rate_delta_ci_95": [
          0.03125,
          0.125
        ],
        "bt_art_rate_treated": 0.1875,
        "n_prompt_pairs": 4,
        "rv_cohens_dz": -0.78073730491261,
        "rv_delta_ci_95": [
          -0.0745482337508298,
          -0.0007554797605678343
        ],
        "rv_delta_mean": -0.036280926494043686,
        "rv_p_value": 0.21633515830204544
      },
      "anchor_early_mlp_0p125_bridge_3": {
        "alpha": 3.125,
        "bt_art_cohens_h": 0.35102142159452576,
        "bt_art_exact_sign_p": 0.25,
        "bt_art_prompt_losses": 0,
        "bt_art_prompt_wins": 3,
        "bt_art_rate_control": 0.09375,
        "bt_art_rate_delta": 0.125,
        "bt_art_rate_delta_ci_95": [
          0.03125,
          0.21875
        ],
        "bt_art_rate_treated": 0.21875,
        "n_prompt_pairs": 4,
        "rv_cohens_dz": -0.030508213200054184,
        "rv_delta_ci_95": [
          -0.0673885605587646,
          0.06134330683335215
        ],
        "rv_delta_mean": -0.0024963560399015594,
        "rv_p_value": 0.9551835754741309
      },
      "anchor_only": {
        "alpha": 0.0,
        "bt_art_cohens_h": 0.0,
        "bt_art_exact_sign_p": null,
        "bt_art_prompt_losses": 0,
        "bt_art_prompt_wins": 0,
        "bt_art_rate_control": 0.09375,
        "bt_art_rate_delta": 0.0,
        "bt_art_rate_delta_ci_95": [
          0.0,
          0.0
        ],
        "bt_art_rate_treated": 0.09375,
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
        "bt_art_cohens_h": 0.2732963053028443,
        "bt_art_exact_sign_p": 0.25,
        "bt_art_prompt_losses": 0,
        "bt_art_prompt_wins": 3,
        "bt_art_rate_control": 0.09375,
        "bt_art_rate_delta": 0.09375,
        "bt_art_rate_delta_ci_95": [
          0.03125,
          0.125
        ],
        "bt_art_rate_treated": 0.1875,
        "n_prompt_pairs": 4,
        "rv_cohens_dz": -0.78073730491261,
        "rv_delta_ci_95": [
          -0.0745482337508298,
          -0.0007554797605678343
        ],
        "rv_delta_mean": -0.036280926494043686,
        "rv_p_value": 0.21633515830204544
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
    606,
    707,
    808
  ],
  "heldout_prompt_counts": {
    "by_group": {
      "baseline_creative": 4,
      "baseline_factual": 4,
      "baseline_math": 4,
      "champions": 4
    },
    "by_mode": {
      "baseline": 12,
      "recursive": 4
    },
    "total": 16
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
  "n_generation_seeds": 8,
  "n_pairs": 16,
  "n_total": 16,
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
  "timestamp": "20260314_140032",
  "top_p": 0.95,
  "verdict": "multisite_sufficient"
}
```
