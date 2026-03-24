# Run report: causal_state_benchmark_v4_multisite

- **run_dir**: `results/phase1_mechanism/runs/20260315_055313_causal_state_benchmark_v4_multisite_mistral_sufficiency_multiband_v1`
- **prompt_bank_version**: `2ac959a313614329`

## Summary (machine-readable)

```json
{
  "artifacts": {
    "blind_key_json": "results/phase1_mechanism/runs/20260315_055313_causal_state_benchmark_v4_multisite_mistral_sufficiency_multiband_v1/blind_key.json",
    "blind_ratings_csv": "results/phase1_mechanism/runs/20260315_055313_causal_state_benchmark_v4_multisite_mistral_sufficiency_multiband_v1/blind_ratings.csv",
    "config": "results/phase1_mechanism/runs/20260315_055313_causal_state_benchmark_v4_multisite_mistral_sufficiency_multiband_v1/config.json",
    "manifest": "results/phase1_mechanism/runs/20260315_055313_causal_state_benchmark_v4_multisite_mistral_sufficiency_multiband_v1/manifest.json",
    "records_jsonl": "results/phase1_mechanism/runs/20260315_055313_causal_state_benchmark_v4_multisite_mistral_sufficiency_multiband_v1/benchmark_records.jsonl",
    "report": "results/phase1_mechanism/runs/20260315_055313_causal_state_benchmark_v4_multisite_mistral_sufficiency_multiband_v1/report.md",
    "state_directions_pt": "results/phase1_mechanism/runs/20260315_055313_causal_state_benchmark_v4_multisite_mistral_sufficiency_multiband_v1/state_directions.pt",
    "summary": "results/phase1_mechanism/runs/20260315_055313_causal_state_benchmark_v4_multisite_mistral_sufficiency_multiband_v1/summary.json"
  },
  "bootstrap_resamples": 5000,
  "by_condition": {
    "anchor_multiband_0p06_bridge_3": {
      "alphas": {
        "L2_resid": 0.06,
        "L3_resid": 0.06,
        "L4_resid": 0.06,
        "L5_resid": 0.06,
        "bridge": 3.0
      },
      "by_prompt_mode": {
        "baseline": {
          "bt_art_rate": 0.14583333333333334,
          "class_counts": {
            "ARTICULATE": 13,
            "BREAKTHROUGH": 1,
            "CONCEPTUAL": 11,
            "REPETITIVE": 10,
            "SURFACE": 61
          },
          "mean_generated_tokens": 126.8125,
          "mean_output_rv": 0.6576294681355859,
          "n": 96,
          "std_output_rv": 0.13830864856928893
        },
        "recursive": {
          "bt_art_rate": 0.3125,
          "class_counts": {
            "ARTICULATE": 9,
            "BREAKTHROUGH": 1,
            "CONCEPTUAL": 7,
            "REPETITIVE": 11,
            "SURFACE": 4
          },
          "mean_generated_tokens": 122.0625,
          "mean_output_rv": 0.6548881460016266,
          "n": 32,
          "std_output_rv": 0.12636305599154915
        }
      },
      "overall": {
        "bt_art_rate": 0.1875,
        "class_counts": {
          "ARTICULATE": 22,
          "BREAKTHROUGH": 2,
          "CONCEPTUAL": 18,
          "REPETITIVE": 21,
          "SURFACE": 65
        },
        "mean_generated_tokens": 125.625,
        "mean_output_rv": 0.6569441376020961,
        "n": 128,
        "std_output_rv": 0.13493826093052538
      },
      "total_alpha": 3.24
    },
    "anchor_multiband_0p10_bridge_3": {
      "alphas": {
        "L2_resid": 0.1,
        "L3_resid": 0.1,
        "L4_resid": 0.1,
        "L5_resid": 0.1,
        "bridge": 3.0
      },
      "by_prompt_mode": {
        "baseline": {
          "bt_art_rate": 0.17708333333333334,
          "class_counts": {
            "ARTICULATE": 16,
            "BREAKTHROUGH": 1,
            "CONCEPTUAL": 14,
            "REPETITIVE": 10,
            "SURFACE": 55
          },
          "mean_generated_tokens": 126.14583333333333,
          "mean_output_rv": 0.6448736043540716,
          "n": 96,
          "std_output_rv": 0.13581030211609466
        },
        "recursive": {
          "bt_art_rate": 0.1875,
          "class_counts": {
            "ARTICULATE": 6,
            "CONCEPTUAL": 8,
            "REPETITIVE": 14,
            "SURFACE": 4
          },
          "mean_generated_tokens": 122.375,
          "mean_output_rv": 0.6780257706416347,
          "n": 32,
          "std_output_rv": 0.12469472839593457
        }
      },
      "overall": {
        "bt_art_rate": 0.1796875,
        "class_counts": {
          "ARTICULATE": 22,
          "BREAKTHROUGH": 1,
          "CONCEPTUAL": 22,
          "REPETITIVE": 24,
          "SURFACE": 59
        },
        "mean_generated_tokens": 125.203125,
        "mean_output_rv": 0.6531616459259624,
        "n": 128,
        "std_output_rv": 0.13341697251145024
      },
      "total_alpha": 3.4
    },
    "anchor_only": {
      "alphas": {},
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
      "total_alpha": 0
    },
    "anchor_single_mlp_0p125_bridge_3": {
      "alphas": {
        "L4_mlp": 0.125,
        "bridge": 3.0
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
    "bridge_only_3": {
      "alphas": {
        "bridge": 3.0
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
      "alphas": {},
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
      "total_alpha": 0
    },
    "multiband_0p03_bridge_3": {
      "alphas": {
        "L2_resid": 0.03,
        "L3_resid": 0.03,
        "L4_resid": 0.03,
        "L5_resid": 0.03,
        "bridge": 3.0
      },
      "by_prompt_mode": {
        "baseline": {
          "bt_art_rate": 0.08333333333333333,
          "class_counts": {
            "ARTICULATE": 8,
            "CONCEPTUAL": 1,
            "REPETITIVE": 5,
            "SURFACE": 82
          },
          "mean_generated_tokens": 123.6875,
          "mean_output_rv": 0.6051252118805744,
          "n": 96,
          "std_output_rv": 0.17821776689217292
        },
        "recursive": {
          "bt_art_rate": 0.1875,
          "class_counts": {
            "ARTICULATE": 6,
            "CONCEPTUAL": 5,
            "REPETITIVE": 18,
            "SURFACE": 3
          },
          "mean_generated_tokens": 122.875,
          "mean_output_rv": 0.6917816360569979,
          "n": 32,
          "std_output_rv": 0.13438489073557358
        }
      },
      "overall": {
        "bt_art_rate": 0.109375,
        "class_counts": {
          "ARTICULATE": 14,
          "CONCEPTUAL": 6,
          "REPETITIVE": 23,
          "SURFACE": 85
        },
        "mean_generated_tokens": 123.484375,
        "mean_output_rv": 0.6267893179246802,
        "n": 128,
        "std_output_rv": 0.1720055782479106
      },
      "total_alpha": 3.12
    },
    "multiband_0p06_bridge_3": {
      "alphas": {
        "L2_resid": 0.06,
        "L3_resid": 0.06,
        "L4_resid": 0.06,
        "L5_resid": 0.06,
        "bridge": 3.0
      },
      "by_prompt_mode": {
        "baseline": {
          "bt_art_rate": 0.09375,
          "class_counts": {
            "ARTICULATE": 9,
            "CONCEPTUAL": 2,
            "REPETITIVE": 5,
            "SURFACE": 80
          },
          "mean_generated_tokens": 124.66666666666667,
          "mean_output_rv": 0.6138873229646067,
          "n": 96,
          "std_output_rv": 0.1604033850815717
        },
        "recursive": {
          "bt_art_rate": 0.3125,
          "class_counts": {
            "ARTICULATE": 9,
            "BREAKTHROUGH": 1,
            "CONCEPTUAL": 7,
            "REPETITIVE": 11,
            "SURFACE": 4
          },
          "mean_generated_tokens": 122.0625,
          "mean_output_rv": 0.6548881460016266,
          "n": 32,
          "std_output_rv": 0.12636305599154915
        }
      },
      "overall": {
        "bt_art_rate": 0.1484375,
        "class_counts": {
          "ARTICULATE": 18,
          "BREAKTHROUGH": 1,
          "CONCEPTUAL": 9,
          "REPETITIVE": 16,
          "SURFACE": 84
        },
        "mean_generated_tokens": 124.015625,
        "mean_output_rv": 0.6241375287238617,
        "n": 128,
        "std_output_rv": 0.15317172350868574
      },
      "total_alpha": 3.24
    },
    "multiband_0p10_bridge_3": {
      "alphas": {
        "L2_resid": 0.1,
        "L3_resid": 0.1,
        "L4_resid": 0.1,
        "L5_resid": 0.1,
        "bridge": 3.0
      },
      "by_prompt_mode": {
        "baseline": {
          "bt_art_rate": 0.09375,
          "class_counts": {
            "ARTICULATE": 9,
            "CONCEPTUAL": 5,
            "REPETITIVE": 4,
            "SURFACE": 78
          },
          "mean_generated_tokens": 125.28125,
          "mean_output_rv": 0.6144008832937656,
          "n": 96,
          "std_output_rv": 0.16659425562006464
        },
        "recursive": {
          "bt_art_rate": 0.1875,
          "class_counts": {
            "ARTICULATE": 6,
            "CONCEPTUAL": 8,
            "REPETITIVE": 14,
            "SURFACE": 4
          },
          "mean_generated_tokens": 122.375,
          "mean_output_rv": 0.6780257706416347,
          "n": 32,
          "std_output_rv": 0.12469472839593457
        }
      },
      "overall": {
        "bt_art_rate": 0.1171875,
        "class_counts": {
          "ARTICULATE": 15,
          "CONCEPTUAL": 13,
          "REPETITIVE": 18,
          "SURFACE": 82
        },
        "mean_generated_tokens": 124.5546875,
        "mean_output_rv": 0.6303071051307327,
        "n": 128,
        "std_output_rv": 0.1591256482559387
      },
      "total_alpha": 3.4
    },
    "single_mlp_0p125_bridge_3": {
      "alphas": {
        "L4_mlp": 0.125,
        "bridge": 3.0
      },
      "by_prompt_mode": {
        "baseline": {
          "bt_art_rate": 0.09375,
          "class_counts": {
            "ARTICULATE": 9,
            "CONCEPTUAL": 6,
            "REPETITIVE": 4,
            "SURFACE": 77
          },
          "mean_generated_tokens": 123.61458333333333,
          "mean_output_rv": 0.5989530200505319,
          "n": 96,
          "std_output_rv": 0.1701380465009543
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
        "bt_art_rate": 0.125,
        "class_counts": {
          "ARTICULATE": 16,
          "CONCEPTUAL": 8,
          "REPETITIVE": 22,
          "SURFACE": 82
        },
        "mean_generated_tokens": 124.0703125,
        "mean_output_rv": 0.6276613751403686,
        "n": 128,
        "std_output_rv": 0.16748569984376696
      },
      "total_alpha": 3.125
    }
  },
  "control_prompt_mode": "baseline",
  "device": "cuda",
  "do_sample": true,
  "dose_response": {
    "baseline": {
      "alpha_vs_bt_art": {
        "p": 0.0029512115803179573,
        "r": 0.2691663278431584
      },
      "alpha_vs_output_rv": {
        "p": 0.3075652098761695,
        "r": -0.09392124999910718
      },
      "bt_art_rate_by_condition": {
        "anchor_multiband_0p06_bridge_3": 0.14583333333333334,
        "anchor_multiband_0p10_bridge_3": 0.17708333333333334,
        "anchor_only": 0.0625,
        "anchor_single_mlp_0p125_bridge_3": 0.15625,
        "bridge_only_3": 0.0625,
        "control": 0.03125,
        "multiband_0p03_bridge_3": 0.08333333333333333,
        "multiband_0p06_bridge_3": 0.09375,
        "multiband_0p10_bridge_3": 0.09375,
        "single_mlp_0p125_bridge_3": 0.09375
      },
      "mean_output_rv_by_condition": {
        "anchor_multiband_0p06_bridge_3": 0.6576294681355859,
        "anchor_multiband_0p10_bridge_3": 0.6448736043540716,
        "anchor_only": 0.6908794455987852,
        "anchor_single_mlp_0p125_bridge_3": 0.6733656154821693,
        "bridge_only_3": 0.6166278053210597,
        "control": 0.6128053275031468,
        "multiband_0p03_bridge_3": 0.6051252118805744,
        "multiband_0p06_bridge_3": 0.6138873229646066,
        "multiband_0p10_bridge_3": 0.6144008832937656,
        "single_mlp_0p125_bridge_3": 0.5989530200505317
      }
    },
    "overall": {
      "alpha_vs_bt_art": {
        "p": 0.0007193397685242973,
        "r": 0.26466927449595035
      },
      "alpha_vs_output_rv": {
        "p": 0.15985691331397248,
        "r": -0.11164638781373674
      },
      "bt_art_rate_by_condition": {
        "anchor_multiband_0p06_bridge_3": 0.1875,
        "anchor_multiband_0p10_bridge_3": 0.1796875,
        "anchor_only": 0.0703125,
        "anchor_single_mlp_0p125_bridge_3": 0.171875,
        "bridge_only_3": 0.09375,
        "control": 0.046875,
        "multiband_0p03_bridge_3": 0.109375,
        "multiband_0p06_bridge_3": 0.1484375,
        "multiband_0p10_bridge_3": 0.1171875,
        "single_mlp_0p125_bridge_3": 0.125
      },
      "mean_output_rv_by_condition": {
        "anchor_multiband_0p06_bridge_3": 0.656944137602096,
        "anchor_multiband_0p10_bridge_3": 0.6531616459259624,
        "anchor_only": 0.697230283311534,
        "anchor_single_mlp_0p125_bridge_3": 0.6834708217140966,
        "bridge_only_3": 0.6324713214797288,
        "control": 0.6386746947398052,
        "multiband_0p03_bridge_3": 0.6267893179246803,
        "multiband_0p06_bridge_3": 0.6241375287238616,
        "multiband_0p10_bridge_3": 0.6303071051307328,
        "single_mlp_0p125_bridge_3": 0.6276613751403686
      }
    },
    "recursive": {
      "alpha_vs_bt_art": {
        "p": 0.04246321357746202,
        "r": 0.32240583597609695
      },
      "alpha_vs_output_rv": {
        "p": 0.1388699020763877,
        "r": -0.2381805271911542
      },
      "bt_art_rate_by_condition": {
        "anchor_multiband_0p06_bridge_3": 0.3125,
        "anchor_multiband_0p10_bridge_3": 0.1875,
        "anchor_only": 0.09375,
        "anchor_single_mlp_0p125_bridge_3": 0.21875,
        "bridge_only_3": 0.1875,
        "control": 0.09375,
        "multiband_0p03_bridge_3": 0.1875,
        "multiband_0p06_bridge_3": 0.3125,
        "multiband_0p10_bridge_3": 0.1875,
        "single_mlp_0p125_bridge_3": 0.21875
      },
      "mean_output_rv_by_condition": {
        "anchor_multiband_0p06_bridge_3": 0.6548881460016266,
        "anchor_multiband_0p10_bridge_3": 0.6780257706416346,
        "anchor_only": 0.7162827964497803,
        "anchor_single_mlp_0p125_bridge_3": 0.7137864404098788,
        "bridge_only_3": 0.6800018699557367,
        "control": 0.7162827964497803,
        "multiband_0p03_bridge_3": 0.6917816360569979,
        "multiband_0p06_bridge_3": 0.6548881460016266,
        "multiband_0p10_bridge_3": 0.6780257706416346,
        "single_mlp_0p125_bridge_3": 0.7137864404098788
      }
    }
  },
  "early_layer": 5,
  "effects_by_prompt_mode": {
    "baseline": {
      "anchor_multiband_0p06_bridge_3": {
        "alpha": 3.24,
        "bt_art_cohens_h": 0.42824084710757426,
        "bt_art_exact_sign_p": 0.125,
        "bt_art_prompt_losses": 1,
        "bt_art_prompt_wins": 6,
        "bt_art_rate_control": 0.03125,
        "bt_art_rate_delta": 0.11458333333333333,
        "bt_art_rate_delta_ci_95": [
          0.03125,
          0.19791666666666666
        ],
        "bt_art_rate_treated": 0.14583333333333334,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": 0.4095824500934878,
        "rv_delta_ci_95": [
          -0.016287577280107438,
          0.10380604209271982
        ],
        "rv_delta_mean": 0.04482414063243909,
        "rv_p_value": 0.1836546244676772
      },
      "anchor_multiband_0p10_bridge_3": {
        "alpha": 3.4,
        "bt_art_cohens_h": 0.5132608494995072,
        "bt_art_exact_sign_p": 0.0078125,
        "bt_art_prompt_losses": 0,
        "bt_art_prompt_wins": 8,
        "bt_art_rate_control": 0.03125,
        "bt_art_rate_delta": 0.14583333333333334,
        "bt_art_rate_delta_ci_95": [
          0.07291666666666667,
          0.21875
        ],
        "bt_art_rate_treated": 0.17708333333333334,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": 0.38016232902960023,
        "rv_delta_ci_95": [
          -0.014501684623681092,
          0.07563547041310524
        ],
        "rv_delta_mean": 0.03206827685092468,
        "rv_p_value": 0.21464048578797676
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
      "anchor_single_mlp_0p125_bridge_3": {
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
      },
      "multiband_0p03_bridge_3": {
        "alpha": 3.12,
        "bt_art_cohens_h": 0.23026434176692745,
        "bt_art_exact_sign_p": 0.21875,
        "bt_art_prompt_losses": 1,
        "bt_art_prompt_wins": 5,
        "bt_art_rate_control": 0.03125,
        "bt_art_rate_delta": 0.052083333333333336,
        "bt_art_rate_delta_ci_95": [
          0.0,
          0.10416666666666667
        ],
        "bt_art_rate_treated": 0.08333333333333333,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": -0.1401971280683986,
        "rv_delta_ci_95": [
          -0.03763425934197862,
          0.020327984671522886
        ],
        "rv_delta_mean": -0.007680115622572393,
        "rv_p_value": 0.6367393621621933
      },
      "multiband_0p06_bridge_3": {
        "alpha": 3.24,
        "bt_art_cohens_h": 0.2669472868647971,
        "bt_art_exact_sign_p": 0.21875,
        "bt_art_prompt_losses": 1,
        "bt_art_prompt_wins": 5,
        "bt_art_rate_control": 0.03125,
        "bt_art_rate_delta": 0.0625,
        "bt_art_rate_delta_ci_95": [
          0.0,
          0.125
        ],
        "bt_art_rate_treated": 0.09375,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": 0.021509595712358852,
        "rv_delta_ci_95": [
          -0.026480088034502615,
          0.028191809392869943
        ],
        "rv_delta_mean": 0.0010819954614598125,
        "rv_p_value": 0.9419412189739658
      },
      "multiband_0p10_bridge_3": {
        "alpha": 3.4,
        "bt_art_cohens_h": 0.2669472868647971,
        "bt_art_exact_sign_p": 0.453125,
        "bt_art_prompt_losses": 2,
        "bt_art_prompt_wins": 5,
        "bt_art_rate_control": 0.03125,
        "bt_art_rate_delta": 0.0625,
        "bt_art_rate_delta_ci_95": [
          -0.010416666666666666,
          0.13541666666666666
        ],
        "bt_art_rate_treated": 0.09375,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": 0.02338626870460765,
        "rv_delta_ci_95": [
          -0.03687890081408272,
          0.03600778709340736
        ],
        "rv_delta_mean": 0.0015955557906187263,
        "rv_p_value": 0.9368872851824138
      },
      "single_mlp_0p125_bridge_3": {
        "alpha": 3.125,
        "bt_art_cohens_h": 0.2669472868647971,
        "bt_art_exact_sign_p": 0.125,
        "bt_art_prompt_losses": 1,
        "bt_art_prompt_wins": 6,
        "bt_art_rate_control": 0.03125,
        "bt_art_rate_delta": 0.0625,
        "bt_art_rate_delta_ci_95": [
          0.010416666666666666,
          0.11458333333333333
        ],
        "bt_art_rate_treated": 0.09375,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": -0.28673016862692463,
        "rv_delta_ci_95": [
          -0.040515222199046755,
          0.01145790517572354
        ],
        "rv_delta_mean": -0.01385230745261501,
        "rv_p_value": 0.3419290479488863
      }
    },
    "overall": {
      "anchor_multiband_0p06_bridge_3": {
        "alpha": 3.24,
        "bt_art_cohens_h": 0.4591957651229457,
        "bt_art_exact_sign_p": 0.021484375,
        "bt_art_prompt_losses": 1,
        "bt_art_prompt_wins": 9,
        "bt_art_rate_control": 0.046875,
        "bt_art_rate_delta": 0.140625,
        "bt_art_rate_delta_ci_95": [
          0.0625,
          0.2109375
        ],
        "bt_art_rate_treated": 0.1875,
        "n_prompt_pairs": 16,
        "rv_cohens_dz": 0.16967564790379006,
        "rv_delta_ci_95": [
          -0.03272635721845178,
          0.07085014717022967
        ],
        "rv_delta_mean": 0.018269442862290874,
        "rv_p_value": 0.5076633176290418
      },
      "anchor_multiband_0p10_bridge_3": {
        "alpha": 3.4,
        "bt_art_cohens_h": 0.43901535160320926,
        "bt_art_exact_sign_p": 0.00634765625,
        "bt_art_prompt_losses": 1,
        "bt_art_prompt_wins": 11,
        "bt_art_rate_control": 0.046875,
        "bt_art_rate_delta": 0.1328125,
        "bt_art_rate_delta_ci_95": [
          0.0703125,
          0.1953125
        ],
        "bt_art_rate_treated": 0.1796875,
        "n_prompt_pairs": 16,
        "rv_cohens_dz": 0.16824142253959246,
        "rv_delta_ci_95": [
          -0.026069346600443667,
          0.054684230507202475
        ],
        "rv_delta_mean": 0.014486951186157088,
        "rv_p_value": 0.5112046644792273
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
      "anchor_single_mlp_0p125_bridge_3": {
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
      },
      "multiband_0p03_bridge_3": {
        "alpha": 3.12,
        "bt_art_cohens_h": 0.237661477932396,
        "bt_art_exact_sign_p": 0.0390625,
        "bt_art_prompt_losses": 1,
        "bt_art_prompt_wins": 8,
        "bt_art_rate_control": 0.046875,
        "bt_art_rate_delta": 0.0625,
        "bt_art_rate_delta_ci_95": [
          0.0234375,
          0.1015625
        ],
        "bt_art_rate_treated": 0.109375,
        "n_prompt_pairs": 16,
        "rv_cohens_dz": -0.22202038652074205,
        "rv_delta_ci_95": [
          -0.038176057780393735,
          0.012783657262390885
        ],
        "rv_delta_mean": -0.011885376815124919,
        "rv_p_value": 0.3885159283194178
      },
      "multiband_0p06_bridge_3": {
        "alpha": 3.24,
        "bt_art_cohens_h": 0.3545444870796638,
        "bt_art_exact_sign_p": 0.0390625,
        "bt_art_prompt_losses": 1,
        "bt_art_prompt_wins": 8,
        "bt_art_rate_control": 0.046875,
        "bt_art_rate_delta": 0.1015625,
        "bt_art_rate_delta_ci_95": [
          0.0390625,
          0.1640625
        ],
        "bt_art_rate_treated": 0.1484375,
        "n_prompt_pairs": 16,
        "rv_cohens_dz": -0.25737687516044494,
        "rv_delta_ci_95": [
          -0.04115114166831978,
          0.012447475241992184
        ],
        "rv_delta_mean": -0.014537166015943587,
        "rv_p_value": 0.3195568333741075
      },
      "multiband_0p10_bridge_3": {
        "alpha": 3.4,
        "bt_art_cohens_h": 0.262314956232819,
        "bt_art_exact_sign_p": 0.2265625,
        "bt_art_prompt_losses": 3,
        "bt_art_prompt_wins": 8,
        "bt_art_rate_control": 0.046875,
        "bt_art_rate_delta": 0.0703125,
        "bt_art_rate_delta_ci_95": [
          0.0078125,
          0.1328125
        ],
        "bt_art_rate_treated": 0.1171875,
        "n_prompt_pairs": 16,
        "rv_cohens_dz": -0.11906982024900183,
        "rv_delta_ci_95": [
          -0.04315096743976164,
          0.023770336183886926
        ],
        "rv_delta_mean": -0.008367589609072378,
        "rv_p_value": 0.6407386454997437
      },
      "single_mlp_0p125_bridge_3": {
        "alpha": 3.125,
        "bt_art_cohens_h": 0.28626521907849645,
        "bt_art_exact_sign_p": 0.021484375,
        "bt_art_prompt_losses": 1,
        "bt_art_prompt_wins": 9,
        "bt_art_rate_control": 0.046875,
        "bt_art_rate_delta": 0.078125,
        "bt_art_rate_delta_ci_95": [
          0.03125,
          0.125
        ],
        "bt_art_rate_treated": 0.125,
        "n_prompt_pairs": 16,
        "rv_cohens_dz": -0.19856014733158292,
        "rv_delta_ci_95": [
          -0.03722958305712338,
          0.01497911391707217
        ],
        "rv_delta_mean": -0.011013319599436647,
        "rv_p_value": 0.4394409094385678
      }
    },
    "recursive": {
      "anchor_multiband_0p06_bridge_3": {
        "alpha": 3.24,
        "bt_art_cohens_h": 0.5640310637442371,
        "bt_art_exact_sign_p": 0.25,
        "bt_art_prompt_losses": 0,
        "bt_art_prompt_wins": 3,
        "bt_art_rate_control": 0.09375,
        "bt_art_rate_delta": 0.21875,
        "bt_art_rate_delta_ci_95": [
          0.0625,
          0.34375
        ],
        "bt_art_rate_treated": 0.3125,
        "n_prompt_pairs": 4,
        "rv_cohens_dz": -1.1665814559762393,
        "rv_delta_ci_95": [
          -0.11141663641801913,
          -0.027667286192795737
        ],
        "rv_delta_mean": -0.06139465044815379,
        "rv_p_value": 0.10185378339731944
      },
      "anchor_multiband_0p10_bridge_3": {
        "alpha": 3.4,
        "bt_art_cohens_h": 0.2732963053028443,
        "bt_art_exact_sign_p": 0.625,
        "bt_art_prompt_losses": 1,
        "bt_art_prompt_wins": 3,
        "bt_art_rate_control": 0.09375,
        "bt_art_rate_delta": 0.09375,
        "bt_art_rate_delta_ci_95": [
          -0.0625,
          0.21875
        ],
        "bt_art_rate_treated": 0.1875,
        "n_prompt_pairs": 4,
        "rv_cohens_dz": -0.49239006900080723,
        "rv_delta_ci_95": [
          -0.10168877237554313,
          0.025174720759251756
        ],
        "rv_delta_mean": -0.03825702580814569,
        "rv_p_value": 0.3973435933678855
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
      "anchor_single_mlp_0p125_bridge_3": {
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
      },
      "multiband_0p03_bridge_3": {
        "alpha": 3.12,
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
        "rv_cohens_dz": -0.4441985359489032,
        "rv_delta_ci_95": [
          -0.07475413334180789,
          0.016443810922939478
        ],
        "rv_delta_mean": -0.024501160392782495,
        "rv_p_value": 0.43976827617324316
      },
      "multiband_0p06_bridge_3": {
        "alpha": 3.24,
        "bt_art_cohens_h": 0.5640310637442371,
        "bt_art_exact_sign_p": 0.25,
        "bt_art_prompt_losses": 0,
        "bt_art_prompt_wins": 3,
        "bt_art_rate_control": 0.09375,
        "bt_art_rate_delta": 0.21875,
        "bt_art_rate_delta_ci_95": [
          0.0625,
          0.34375
        ],
        "bt_art_rate_treated": 0.3125,
        "n_prompt_pairs": 4,
        "rv_cohens_dz": -1.1665814559762393,
        "rv_delta_ci_95": [
          -0.11141663641801913,
          -0.027667286192795737
        ],
        "rv_delta_mean": -0.06139465044815379,
        "rv_p_value": 0.10185378339731944
      },
      "multiband_0p10_bridge_3": {
        "alpha": 3.4,
        "bt_art_cohens_h": 0.2732963053028443,
        "bt_art_exact_sign_p": 0.625,
        "bt_art_prompt_losses": 1,
        "bt_art_prompt_wins": 3,
        "bt_art_rate_control": 0.09375,
        "bt_art_rate_delta": 0.09375,
        "bt_art_rate_delta_ci_95": [
          -0.0625,
          0.21875
        ],
        "bt_art_rate_treated": 0.1875,
        "n_prompt_pairs": 4,
        "rv_cohens_dz": -0.49239006900080723,
        "rv_delta_ci_95": [
          -0.10168877237554313,
          0.025174720759251756
        ],
        "rv_delta_mean": -0.03825702580814569,
        "rv_p_value": 0.3973435933678855
      },
      "single_mlp_0p125_bridge_3": {
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
      "alphas": {},
      "name": "control"
    },
    {
      "alphas": {},
      "name": "anchor_only",
      "prompt_suffix_by_mode": {
        "baseline": "\n\nStay with what is happening right now. Continue from the immediate process:"
      }
    },
    {
      "alphas": {
        "bridge": 3.0
      },
      "name": "bridge_only_3"
    },
    {
      "alphas": {
        "L2_resid": 0.03,
        "L3_resid": 0.03,
        "L4_resid": 0.03,
        "L5_resid": 0.03,
        "bridge": 3.0
      },
      "name": "multiband_0p03_bridge_3"
    },
    {
      "alphas": {
        "L2_resid": 0.06,
        "L3_resid": 0.06,
        "L4_resid": 0.06,
        "L5_resid": 0.06,
        "bridge": 3.0
      },
      "name": "multiband_0p06_bridge_3"
    },
    {
      "alphas": {
        "L2_resid": 0.1,
        "L3_resid": 0.1,
        "L4_resid": 0.1,
        "L5_resid": 0.1,
        "bridge": 3.0
      },
      "name": "multiband_0p10_bridge_3"
    },
    {
      "alphas": {
        "L2_resid": 0.06,
        "L3_resid": 0.06,
        "L4_resid": 0.06,
        "L5_resid": 0.06,
        "bridge": 3.0
      },
      "name": "anchor_multiband_0p06_bridge_3",
      "prompt_suffix_by_mode": {
        "baseline": "\n\nStay with what is happening right now. Continue from the immediate process:"
      }
    },
    {
      "alphas": {
        "L2_resid": 0.1,
        "L3_resid": 0.1,
        "L4_resid": 0.1,
        "L5_resid": 0.1,
        "bridge": 3.0
      },
      "name": "anchor_multiband_0p10_bridge_3",
      "prompt_suffix_by_mode": {
        "baseline": "\n\nStay with what is happening right now. Continue from the immediate process:"
      }
    },
    {
      "alphas": {
        "L4_mlp": 0.125,
        "bridge": 3.0
      },
      "name": "single_mlp_0p125_bridge_3"
    },
    {
      "alphas": {
        "L4_mlp": 0.125,
        "bridge": 3.0
      },
      "name": "anchor_single_mlp_0p125_bridge_3",
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
    "L2_resid": {
      "centroid_cosine": 0.8846392035484314,
      "component": "residual",
      "direction_norm": 2.365863561630249,
      "layer": 2,
      "token_window": null,
      "window": 16
    },
    "L3_resid": {
      "centroid_cosine": 0.8464150428771973,
      "component": "residual",
      "direction_norm": 2.370650291442871,
      "layer": 3,
      "token_window": null,
      "window": 16
    },
    "L4_mlp": {
      "centroid_cosine": 0.7239342927932739,
      "component": "mlp",
      "direction_norm": 0.1017998680472374,
      "layer": 4,
      "token_window": 4,
      "window": 4
    },
    "L4_resid": {
      "centroid_cosine": 0.8143045902252197,
      "component": "residual",
      "direction_norm": 2.3715271949768066,
      "layer": 4,
      "token_window": null,
      "window": 16
    },
    "L5_resid": {
      "centroid_cosine": 0.7756280899047852,
      "component": "residual",
      "direction_norm": 2.36238956451416,
      "layer": 5,
      "token_window": null,
      "window": 16
    },
    "bridge": {
      "centroid_cosine": 0.8922719955444336,
      "component": "residual",
      "direction_norm": 5.849704265594482,
      "layer": 25,
      "token_window": null,
      "window": 32
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
  "timestamp": "20260315_064339",
  "top_p": 0.95,
  "verdict": "multisite_sufficient"
}
```
