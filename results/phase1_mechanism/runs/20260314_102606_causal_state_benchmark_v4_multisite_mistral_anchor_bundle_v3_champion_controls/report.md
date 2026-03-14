# Run report: causal_state_benchmark_v4_multisite

- **run_dir**: `results/phase1_mechanism/runs/20260314_102606_causal_state_benchmark_v4_multisite_mistral_anchor_bundle_v3_champion_controls`
- **prompt_bank_version**: `2ac959a313614329`

## Summary (machine-readable)

```json
{
  "artifacts": {
    "blind_key_json": "results/phase1_mechanism/runs/20260314_102606_causal_state_benchmark_v4_multisite_mistral_anchor_bundle_v3_champion_controls/blind_key.json",
    "blind_ratings_csv": "results/phase1_mechanism/runs/20260314_102606_causal_state_benchmark_v4_multisite_mistral_anchor_bundle_v3_champion_controls/blind_ratings.csv",
    "config": "results/phase1_mechanism/runs/20260314_102606_causal_state_benchmark_v4_multisite_mistral_anchor_bundle_v3_champion_controls/config.json",
    "manifest": "results/phase1_mechanism/runs/20260314_102606_causal_state_benchmark_v4_multisite_mistral_anchor_bundle_v3_champion_controls/manifest.json",
    "records_jsonl": "results/phase1_mechanism/runs/20260314_102606_causal_state_benchmark_v4_multisite_mistral_anchor_bundle_v3_champion_controls/benchmark_records.jsonl",
    "report": "results/phase1_mechanism/runs/20260314_102606_causal_state_benchmark_v4_multisite_mistral_anchor_bundle_v3_champion_controls/report.md",
    "state_directions_pt": "results/phase1_mechanism/runs/20260314_102606_causal_state_benchmark_v4_multisite_mistral_anchor_bundle_v3_champion_controls/state_directions.pt",
    "summary": "results/phase1_mechanism/runs/20260314_102606_causal_state_benchmark_v4_multisite_mistral_anchor_bundle_v3_champion_controls/summary.json"
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
          "bt_art_rate": 0.4895833333333333,
          "class_counts": {
            "ARTICULATE": 39,
            "BREAKTHROUGH": 8,
            "CONCEPTUAL": 14,
            "REPETITIVE": 2,
            "SURFACE": 33
          },
          "mean_generated_tokens": 126.33333333333333,
          "mean_output_rv": 0.6116389648676375,
          "n": 96,
          "std_output_rv": 0.09814332482210258
        },
        "recursive": {
          "bt_art_rate": 0.10416666666666667,
          "class_counts": {
            "ARTICULATE": 5,
            "CONCEPTUAL": 7,
            "REPETITIVE": 30,
            "SURFACE": 6
          },
          "mean_generated_tokens": 124.79166666666667,
          "mean_output_rv": 0.6884819730673343,
          "n": 48,
          "std_output_rv": 0.15260386827757455
        }
      },
      "overall": {
        "bt_art_rate": 0.3611111111111111,
        "class_counts": {
          "ARTICULATE": 44,
          "BREAKTHROUGH": 8,
          "CONCEPTUAL": 21,
          "REPETITIVE": 32,
          "SURFACE": 39
        },
        "mean_generated_tokens": 125.81944444444444,
        "mean_output_rv": 0.6372533009342031,
        "n": 144,
        "std_output_rv": 0.12399354262443787
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
          "bt_art_rate": 0.5416666666666666,
          "class_counts": {
            "ARTICULATE": 44,
            "BREAKTHROUGH": 8,
            "CONCEPTUAL": 8,
            "REPETITIVE": 12,
            "SURFACE": 24
          },
          "mean_generated_tokens": 126.67708333333333,
          "mean_output_rv": 0.6176864556997813,
          "n": 96,
          "std_output_rv": 0.13966142390542294
        },
        "recursive": {
          "bt_art_rate": 0.16666666666666666,
          "class_counts": {
            "ARTICULATE": 8,
            "CONCEPTUAL": 1,
            "REPETITIVE": 31,
            "SURFACE": 8
          },
          "mean_generated_tokens": 125.625,
          "mean_output_rv": 0.6832029982329212,
          "n": 48,
          "std_output_rv": 0.14857268213761754
        }
      },
      "overall": {
        "bt_art_rate": 0.4166666666666667,
        "class_counts": {
          "ARTICULATE": 52,
          "BREAKTHROUGH": 8,
          "CONCEPTUAL": 9,
          "REPETITIVE": 43,
          "SURFACE": 32
        },
        "mean_generated_tokens": 126.32638888888889,
        "mean_output_rv": 0.6395253032108279,
        "n": 144,
        "std_output_rv": 0.14551170114663547
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
          "bt_art_rate": 0.375,
          "class_counts": {
            "ARTICULATE": 32,
            "BREAKTHROUGH": 4,
            "CONCEPTUAL": 20,
            "REPETITIVE": 21,
            "SURFACE": 19
          },
          "mean_generated_tokens": 126.27083333333333,
          "mean_output_rv": 0.6163129489498073,
          "n": 96,
          "std_output_rv": 0.10504406494991274
        },
        "recursive": {
          "bt_art_rate": 0.0625,
          "class_counts": {
            "ARTICULATE": 3,
            "CONCEPTUAL": 4,
            "REPETITIVE": 34,
            "SURFACE": 7
          },
          "mean_generated_tokens": 126.72916666666667,
          "mean_output_rv": 0.6919750772653881,
          "n": 48,
          "std_output_rv": 0.1369620021351995
        }
      },
      "overall": {
        "bt_art_rate": 0.2708333333333333,
        "class_counts": {
          "ARTICULATE": 35,
          "BREAKTHROUGH": 4,
          "CONCEPTUAL": 24,
          "REPETITIVE": 55,
          "SURFACE": 26
        },
        "mean_generated_tokens": 126.42361111111111,
        "mean_output_rv": 0.6415336583883343,
        "n": 144,
        "std_output_rv": 0.12156037541998449
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
          "bt_art_rate": 0.08333333333333333,
          "class_counts": {
            "ARTICULATE": 7,
            "BREAKTHROUGH": 1,
            "CONCEPTUAL": 1,
            "REPETITIVE": 73,
            "SURFACE": 14
          },
          "mean_generated_tokens": 128.0,
          "mean_output_rv": 0.9064753152859307,
          "n": 96,
          "std_output_rv": 0.19873958818911355
        },
        "recursive": {
          "bt_art_rate": 0.10416666666666667,
          "class_counts": {
            "ARTICULATE": 5,
            "CONCEPTUAL": 7,
            "REPETITIVE": 30,
            "SURFACE": 6
          },
          "mean_generated_tokens": 124.79166666666667,
          "mean_output_rv": 0.6884819730673343,
          "n": 48,
          "std_output_rv": 0.15260386827757455
        }
      },
      "overall": {
        "bt_art_rate": 0.09027777777777778,
        "class_counts": {
          "ARTICULATE": 12,
          "BREAKTHROUGH": 1,
          "CONCEPTUAL": 8,
          "REPETITIVE": 103,
          "SURFACE": 20
        },
        "mean_generated_tokens": 126.93055555555556,
        "mean_output_rv": 0.833810867879732,
        "n": 144,
        "std_output_rv": 0.21101591748309745
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
          "bt_art_rate": 0.052083333333333336,
          "class_counts": {
            "ARTICULATE": 4,
            "BREAKTHROUGH": 1,
            "CONCEPTUAL": 3,
            "REPETITIVE": 74,
            "SURFACE": 14
          },
          "mean_generated_tokens": 128.0,
          "mean_output_rv": 0.9221209321330562,
          "n": 96,
          "std_output_rv": 0.17684019724997233
        },
        "recursive": {
          "bt_art_rate": 0.0625,
          "class_counts": {
            "ARTICULATE": 3,
            "CONCEPTUAL": 4,
            "REPETITIVE": 34,
            "SURFACE": 7
          },
          "mean_generated_tokens": 126.72916666666667,
          "mean_output_rv": 0.6919750772653881,
          "n": 48,
          "std_output_rv": 0.1369620021351995
        }
      },
      "overall": {
        "bt_art_rate": 0.05555555555555555,
        "class_counts": {
          "ARTICULATE": 7,
          "BREAKTHROUGH": 1,
          "CONCEPTUAL": 7,
          "REPETITIVE": 108,
          "SURFACE": 21
        },
        "mean_generated_tokens": 127.57638888888889,
        "mean_output_rv": 0.8454056471771669,
        "n": 144,
        "std_output_rv": 0.19696090725837392
      },
      "total_alpha": 0.0
    },
    "early_mlp_0p125_bridge_3": {
      "alphas": {
        "bridge": 3.0,
        "early_mlp": 0.125
      },
      "by_prompt_mode": {
        "baseline": {
          "bt_art_rate": 0.041666666666666664,
          "class_counts": {
            "ARTICULATE": 3,
            "BREAKTHROUGH": 1,
            "CONCEPTUAL": 5,
            "REPETITIVE": 73,
            "SURFACE": 14
          },
          "mean_generated_tokens": 128.0,
          "mean_output_rv": 0.9121151417039991,
          "n": 96,
          "std_output_rv": 0.19306641441351807
        },
        "recursive": {
          "bt_art_rate": 0.16666666666666666,
          "class_counts": {
            "ARTICULATE": 8,
            "CONCEPTUAL": 1,
            "REPETITIVE": 31,
            "SURFACE": 8
          },
          "mean_generated_tokens": 125.625,
          "mean_output_rv": 0.6832029982329212,
          "n": 48,
          "std_output_rv": 0.14857268213761754
        }
      },
      "overall": {
        "bt_art_rate": 0.08333333333333333,
        "class_counts": {
          "ARTICULATE": 11,
          "BREAKTHROUGH": 1,
          "CONCEPTUAL": 6,
          "REPETITIVE": 104,
          "SURFACE": 22
        },
        "mean_generated_tokens": 127.20833333333333,
        "mean_output_rv": 0.8358110938803066,
        "n": 144,
        "std_output_rv": 0.20915056351950181
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
        "p": 0.27254736155941073,
        "r": 0.13104292505658247
      },
      "alpha_vs_output_rv": {
        "p": 0.8880793235383104,
        "r": -0.016880021865266562
      },
      "bt_art_rate_by_condition": {
        "anchor_bridge_3": 0.4895833333333333,
        "anchor_early_mlp_0p125_bridge_3": 0.5416666666666666,
        "anchor_only": 0.375,
        "bridge_only_3": 0.08333333333333333,
        "control": 0.052083333333333336,
        "early_mlp_0p125_bridge_3": 0.041666666666666664
      },
      "mean_output_rv_by_condition": {
        "anchor_bridge_3": 0.6116389648676376,
        "anchor_early_mlp_0p125_bridge_3": 0.6176864556997813,
        "anchor_only": 0.6163129489498073,
        "bridge_only_3": 0.906475315285931,
        "control": 0.9221209321330562,
        "early_mlp_0p125_bridge_3": 0.9121151417039992
      }
    },
    "overall": {
      "alpha_vs_bt_art": {
        "p": 0.13703672531026154,
        "r": 0.14400744777681637
      },
      "alpha_vs_output_rv": {
        "p": 0.8437056086642845,
        "r": -0.019192912502325396
      },
      "bt_art_rate_by_condition": {
        "anchor_bridge_3": 0.3611111111111111,
        "anchor_early_mlp_0p125_bridge_3": 0.4166666666666667,
        "anchor_only": 0.2708333333333333,
        "bridge_only_3": 0.09027777777777778,
        "control": 0.05555555555555555,
        "early_mlp_0p125_bridge_3": 0.08333333333333333
      },
      "mean_output_rv_by_condition": {
        "anchor_bridge_3": 0.6372533009342031,
        "anchor_early_mlp_0p125_bridge_3": 0.639525303210828,
        "anchor_only": 0.6415336583883343,
        "bridge_only_3": 0.8338108678797319,
        "control": 0.8454056471771669,
        "early_mlp_0p125_bridge_3": 0.8358110938803066
      }
    },
    "recursive": {
      "alpha_vs_bt_art": {
        "p": 0.13117946339913292,
        "r": 0.2563996337408014
      },
      "alpha_vs_output_rv": {
        "p": 0.7109799656292892,
        "r": -0.06395035544429088
      },
      "bt_art_rate_by_condition": {
        "anchor_bridge_3": 0.10416666666666667,
        "anchor_early_mlp_0p125_bridge_3": 0.16666666666666666,
        "anchor_only": 0.0625,
        "bridge_only_3": 0.10416666666666667,
        "control": 0.0625,
        "early_mlp_0p125_bridge_3": 0.16666666666666666
      },
      "mean_output_rv_by_condition": {
        "anchor_bridge_3": 0.6884819730673343,
        "anchor_early_mlp_0p125_bridge_3": 0.6832029982329212,
        "anchor_only": 0.691975077265388,
        "bridge_only_3": 0.6884819730673343,
        "control": 0.691975077265388,
        "early_mlp_0p125_bridge_3": 0.6832029982329212
      }
    }
  },
  "early_layer": 5,
  "effects_by_prompt_mode": {
    "baseline": {
      "anchor_bridge_3": {
        "alpha": 3.0,
        "bt_art_cohens_h": 1.0894680610671887,
        "bt_art_exact_sign_p": 0.00048828125,
        "bt_art_prompt_losses": 0,
        "bt_art_prompt_wins": 12,
        "bt_art_rate_control": 0.052083333333333336,
        "bt_art_rate_delta": 0.4375,
        "bt_art_rate_delta_ci_95": [
          0.34375,
          0.53125
        ],
        "bt_art_rate_treated": 0.4895833333333333,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": -1.7422532913944824,
        "rv_delta_ci_95": [
          -0.40289940562635396,
          -0.2054511306268475
        ],
        "rv_delta_mean": -0.3104819672654189,
        "rv_p_value": 8.485110358365732e-05
      },
      "anchor_early_mlp_0p125_bridge_3": {
        "alpha": 3.125,
        "bt_art_cohens_h": 1.193732988346452,
        "bt_art_exact_sign_p": 0.00048828125,
        "bt_art_prompt_losses": 0,
        "bt_art_prompt_wins": 12,
        "bt_art_rate_control": 0.052083333333333336,
        "bt_art_rate_delta": 0.4895833333333333,
        "bt_art_rate_delta_ci_95": [
          0.375,
          0.6041666666666666
        ],
        "bt_art_rate_treated": 0.5416666666666666,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": -1.994871654240602,
        "rv_delta_ci_95": [
          -0.3815180140218689,
          -0.21634719382022435
        ],
        "rv_delta_mean": -0.3044344764332751,
        "rv_p_value": 2.5522434365075562e-05
      },
      "anchor_only": {
        "alpha": 0.0,
        "bt_art_cohens_h": 0.8576226465937583,
        "bt_art_exact_sign_p": 0.00048828125,
        "bt_art_prompt_losses": 0,
        "bt_art_prompt_wins": 12,
        "bt_art_rate_control": 0.052083333333333336,
        "bt_art_rate_delta": 0.3229166666666667,
        "bt_art_rate_delta_ci_95": [
          0.2604166666666667,
          0.3854166666666667
        ],
        "bt_art_rate_treated": 0.375,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": -1.7404045480231352,
        "rv_delta_ci_95": [
          -0.3925104878418871,
          -0.2017866900847323
        ],
        "rv_delta_mean": -0.30580798318324903,
        "rv_p_value": 8.56326750053366e-05
      },
      "bridge_only_3": {
        "alpha": 3.0,
        "bt_art_cohens_h": 0.12519211839809152,
        "bt_art_exact_sign_p": 0.5,
        "bt_art_prompt_losses": 0,
        "bt_art_prompt_wins": 2,
        "bt_art_rate_control": 0.052083333333333336,
        "bt_art_rate_delta": 0.03125,
        "bt_art_rate_delta_ci_95": [
          0.0,
          0.08333333333333333
        ],
        "bt_art_rate_treated": 0.08333333333333333,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": -0.4945025706736793,
        "rv_delta_ci_95": [
          -0.03450549375191353,
          0.0
        ],
        "rv_delta_mean": -0.015645616847125465,
        "rv_p_value": 0.1147175110693082
      },
      "early_mlp_0p125_bridge_3": {
        "alpha": 3.125,
        "bt_art_cohens_h": -0.04935556273671171,
        "bt_art_exact_sign_p": 1.0,
        "bt_art_prompt_losses": 1,
        "bt_art_prompt_wins": 1,
        "bt_art_rate_control": 0.052083333333333336,
        "bt_art_rate_delta": -0.010416666666666666,
        "bt_art_rate_delta_ci_95": [
          -0.0625,
          0.03125
        ],
        "bt_art_rate_treated": 0.041666666666666664,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": -0.31138913169569227,
        "rv_delta_ci_95": [
          -0.029963746110300855,
          0.0015844186371221276
        ],
        "rv_delta_mean": -0.010005790429057118,
        "rv_p_value": 0.3038142378115518
      }
    },
    "overall": {
      "anchor_bridge_3": {
        "alpha": 3.0,
        "bt_art_cohens_h": 0.8134340039036352,
        "bt_art_exact_sign_p": 0.0009765625,
        "bt_art_prompt_losses": 1,
        "bt_art_prompt_wins": 14,
        "bt_art_rate_control": 0.05555555555555555,
        "bt_art_rate_delta": 0.3055555555555556,
        "bt_art_rate_delta_ci_95": [
          0.19444444444444445,
          0.4166666666666667
        ],
        "bt_art_rate_treated": 0.3611111111111111,
        "n_prompt_pairs": 18,
        "rv_cohens_dz": -1.002503960600478,
        "rv_delta_ci_95": [
          -0.3015157701578896,
          -0.11587469339120354
        ],
        "rv_delta_mean": -0.20815234624296386,
        "rv_p_value": 0.0005363349394795328
      },
      "anchor_early_mlp_0p125_bridge_3": {
        "alpha": 3.125,
        "bt_art_cohens_h": 0.9274659979147908,
        "bt_art_exact_sign_p": 0.000518798828125,
        "bt_art_prompt_losses": 1,
        "bt_art_prompt_wins": 15,
        "bt_art_rate_control": 0.05555555555555555,
        "bt_art_rate_delta": 0.3611111111111111,
        "bt_art_rate_delta_ci_95": [
          0.24305555555555555,
          0.4861111111111111
        ],
        "bt_art_rate_treated": 0.4166666666666667,
        "n_prompt_pairs": 18,
        "rv_cohens_dz": -1.074085981606196,
        "rv_delta_ci_95": [
          -0.2901678727315765,
          -0.1187364947874402
        ],
        "rv_delta_mean": -0.20588034396633895,
        "rv_p_value": 0.0002795741006428568
      },
      "anchor_only": {
        "alpha": 0.0,
        "bt_art_cohens_h": 0.6187950162226836,
        "bt_art_exact_sign_p": 0.00048828125,
        "bt_art_prompt_losses": 0,
        "bt_art_prompt_wins": 12,
        "bt_art_rate_control": 0.05555555555555555,
        "bt_art_rate_delta": 0.2152777777777778,
        "bt_art_rate_delta_ci_95": [
          0.13194444444444445,
          0.2916666666666667
        ],
        "bt_art_rate_treated": 0.2708333333333333,
        "n_prompt_pairs": 18,
        "rv_cohens_dz": -0.9950084478040773,
        "rv_delta_ci_95": [
          -0.29841221687289016,
          -0.11079690673829301
        ],
        "rv_delta_mean": -0.2038719887888327,
        "rv_p_value": 0.0005743985934162964
      },
      "bridge_only_3": {
        "alpha": 3.0,
        "bt_art_cohens_h": 0.13447301869275002,
        "bt_art_exact_sign_p": 0.375,
        "bt_art_prompt_losses": 1,
        "bt_art_prompt_wins": 4,
        "bt_art_rate_control": 0.05555555555555555,
        "bt_art_rate_delta": 0.034722222222222224,
        "bt_art_rate_delta_ci_95": [
          -0.006944444444444444,
          0.0763888888888889
        ],
        "bt_art_rate_treated": 0.09027777777777778,
        "n_prompt_pairs": 18,
        "rv_cohens_dz": -0.35447477008303413,
        "rv_delta_ci_95": [
          -0.027036063715486062,
          0.0022056969124761043
        ],
        "rv_delta_mean": -0.011594779297434866,
        "rv_p_value": 0.1509556555089761
      },
      "early_mlp_0p125_bridge_3": {
        "alpha": 3.125,
        "bt_art_cohens_h": 0.10980329379673442,
        "bt_art_exact_sign_p": 0.6875,
        "bt_art_prompt_losses": 2,
        "bt_art_prompt_wins": 4,
        "bt_art_rate_control": 0.05555555555555555,
        "bt_art_rate_delta": 0.027777777777777776,
        "bt_art_rate_delta_ci_95": [
          -0.027777777777777776,
          0.08333333333333333
        ],
        "bt_art_rate_treated": 0.08333333333333333,
        "n_prompt_pairs": 18,
        "rv_cohens_dz": -0.22796461448256888,
        "rv_delta_ci_95": [
          -0.029035318636523085,
          0.00881244410276663
        ],
        "rv_delta_mean": -0.009594553296860344,
        "rv_p_value": 0.3470171586921237
      }
    },
    "recursive": {
      "anchor_bridge_3": {
        "alpha": 3.0,
        "bt_art_cohens_h": 0.15190364296135894,
        "bt_art_exact_sign_p": 1.0,
        "bt_art_prompt_losses": 1,
        "bt_art_prompt_wins": 2,
        "bt_art_rate_control": 0.0625,
        "bt_art_rate_delta": 0.041666666666666664,
        "bt_art_rate_delta_ci_95": [
          -0.041666666666666664,
          0.14583333333333334
        ],
        "bt_art_rate_treated": 0.10416666666666667,
        "n_prompt_pairs": 6,
        "rv_cohens_dz": -0.0962417543774678,
        "rv_delta_ci_95": [
          -0.027319184335932983,
          0.024365133048125782
        ],
        "rv_delta_mean": -0.0034931041980536723,
        "rv_p_value": 0.8229837791224259
      },
      "anchor_early_mlp_0p125_bridge_3": {
        "alpha": 3.125,
        "bt_art_cohens_h": 0.3357081602837729,
        "bt_art_exact_sign_p": 0.625,
        "bt_art_prompt_losses": 1,
        "bt_art_prompt_wins": 3,
        "bt_art_rate_control": 0.0625,
        "bt_art_rate_delta": 0.10416666666666667,
        "bt_art_rate_delta_ci_95": [
          -0.020833333333333332,
          0.20833333333333334
        ],
        "bt_art_rate_treated": 0.16666666666666666,
        "n_prompt_pairs": 6,
        "rv_cohens_dz": -0.14324775505398643,
        "rv_delta_ci_95": [
          -0.05317772744473467,
          0.03488369564154448
        ],
        "rv_delta_mean": -0.008772079032466795,
        "rv_p_value": 0.7399756393002693
      },
      "anchor_only": {
        "alpha": 0.0,
        "bt_art_cohens_h": 0.0,
        "bt_art_exact_sign_p": null,
        "bt_art_prompt_losses": 0,
        "bt_art_prompt_wins": 0,
        "bt_art_rate_control": 0.0625,
        "bt_art_rate_delta": 0.0,
        "bt_art_rate_delta_ci_95": [
          0.0,
          0.0
        ],
        "bt_art_rate_treated": 0.0625,
        "n_prompt_pairs": 6,
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
        "bt_art_cohens_h": 0.15190364296135894,
        "bt_art_exact_sign_p": 1.0,
        "bt_art_prompt_losses": 1,
        "bt_art_prompt_wins": 2,
        "bt_art_rate_control": 0.0625,
        "bt_art_rate_delta": 0.041666666666666664,
        "bt_art_rate_delta_ci_95": [
          -0.041666666666666664,
          0.14583333333333334
        ],
        "bt_art_rate_treated": 0.10416666666666667,
        "n_prompt_pairs": 6,
        "rv_cohens_dz": -0.0962417543774678,
        "rv_delta_ci_95": [
          -0.027319184335932983,
          0.024365133048125782
        ],
        "rv_delta_mean": -0.0034931041980536723,
        "rv_p_value": 0.8229837791224259
      },
      "early_mlp_0p125_bridge_3": {
        "alpha": 3.125,
        "bt_art_cohens_h": 0.3357081602837729,
        "bt_art_exact_sign_p": 0.625,
        "bt_art_prompt_losses": 1,
        "bt_art_prompt_wins": 3,
        "bt_art_rate_control": 0.0625,
        "bt_art_rate_delta": 0.10416666666666667,
        "bt_art_rate_delta_ci_95": [
          -0.020833333333333332,
          0.20833333333333334
        ],
        "bt_art_rate_treated": 0.16666666666666666,
        "n_prompt_pairs": 6,
        "rv_cohens_dz": -0.14324775505398643,
        "rv_delta_ci_95": [
          -0.05317772744473467,
          0.03488369564154448
        ],
        "rv_delta_mean": -0.008772079032466795,
        "rv_p_value": 0.7399756393002693
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
      "champions": 6,
      "control_length_matched": 6,
      "control_pseudo_recursive": 6
    },
    "by_mode": {
      "baseline": 12,
      "recursive": 6
    },
    "total": 18
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
        "early_mlp": 0.125
      },
      "name": "early_mlp_0p125_bridge_3"
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
  "n_pairs": 18,
  "n_total": 18,
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
  "timestamp": "20260314_105919",
  "top_p": 0.95,
  "verdict": "multisite_sufficient"
}
```
