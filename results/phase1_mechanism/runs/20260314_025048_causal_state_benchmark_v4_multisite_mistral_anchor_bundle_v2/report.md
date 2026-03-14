# Run report: causal_state_benchmark_v4_multisite

- **run_dir**: `results/phase1_mechanism/runs/20260314_025048_causal_state_benchmark_v4_multisite_mistral_anchor_bundle_v2`
- **prompt_bank_version**: `2ac959a313614329`

## Summary (machine-readable)

```json
{
  "artifacts": {
    "blind_key_json": "results/phase1_mechanism/runs/20260314_025048_causal_state_benchmark_v4_multisite_mistral_anchor_bundle_v2/blind_key.json",
    "blind_ratings_csv": "results/phase1_mechanism/runs/20260314_025048_causal_state_benchmark_v4_multisite_mistral_anchor_bundle_v2/blind_ratings.csv",
    "config": "results/phase1_mechanism/runs/20260314_025048_causal_state_benchmark_v4_multisite_mistral_anchor_bundle_v2/config.json",
    "manifest": "results/phase1_mechanism/runs/20260314_025048_causal_state_benchmark_v4_multisite_mistral_anchor_bundle_v2/manifest.json",
    "records_jsonl": "results/phase1_mechanism/runs/20260314_025048_causal_state_benchmark_v4_multisite_mistral_anchor_bundle_v2/benchmark_records.jsonl",
    "report": "results/phase1_mechanism/runs/20260314_025048_causal_state_benchmark_v4_multisite_mistral_anchor_bundle_v2/report.md",
    "state_directions_pt": "results/phase1_mechanism/runs/20260314_025048_causal_state_benchmark_v4_multisite_mistral_anchor_bundle_v2/state_directions.pt",
    "summary": "results/phase1_mechanism/runs/20260314_025048_causal_state_benchmark_v4_multisite_mistral_anchor_bundle_v2/summary.json"
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
          "bt_art_rate": 0.1527777777777778,
          "class_counts": {
            "ARTICULATE": 19,
            "BREAKTHROUGH": 3,
            "CONCEPTUAL": 27,
            "REPETITIVE": 13,
            "SURFACE": 82
          },
          "mean_generated_tokens": 125.94444444444444,
          "mean_output_rv": 0.6567736331687654,
          "n": 144,
          "std_output_rv": 0.13470253487542225
        },
        "recursive": {
          "bt_art_rate": 0.5555555555555556,
          "class_counts": {
            "ARTICULATE": 74,
            "BREAKTHROUGH": 6,
            "CONCEPTUAL": 29,
            "REPETITIVE": 17,
            "SURFACE": 18
          },
          "mean_generated_tokens": 124.34722222222223,
          "mean_output_rv": 0.6188870927883419,
          "n": 144,
          "std_output_rv": 0.108023007619091
        }
      },
      "overall": {
        "bt_art_rate": 0.3541666666666667,
        "class_counts": {
          "ARTICULATE": 93,
          "BREAKTHROUGH": 9,
          "CONCEPTUAL": 56,
          "REPETITIVE": 30,
          "SURFACE": 100
        },
        "mean_generated_tokens": 125.14583333333333,
        "mean_output_rv": 0.6378303629785536,
        "n": 288,
        "std_output_rv": 0.12334921582519282
      },
      "total_alpha": 3.0
    },
    "anchor_early_mlp_0p125_bridge_2": {
      "alphas": {
        "bridge": 2.0,
        "early_mlp": 0.125
      },
      "by_prompt_mode": {
        "baseline": {
          "bt_art_rate": 0.10416666666666667,
          "class_counts": {
            "ARTICULATE": 12,
            "BREAKTHROUGH": 3,
            "CONCEPTUAL": 24,
            "REPETITIVE": 16,
            "SURFACE": 89
          },
          "mean_generated_tokens": 126.9375,
          "mean_output_rv": 0.6914742309884188,
          "n": 144,
          "std_output_rv": 0.14961484785174178
        },
        "recursive": {
          "bt_art_rate": 0.5138888888888888,
          "class_counts": {
            "ARTICULATE": 67,
            "BREAKTHROUGH": 7,
            "CONCEPTUAL": 25,
            "REPETITIVE": 25,
            "SURFACE": 20
          },
          "mean_generated_tokens": 124.24305555555556,
          "mean_output_rv": 0.6124480127233038,
          "n": 144,
          "std_output_rv": 0.12975432565428718
        }
      },
      "overall": {
        "bt_art_rate": 0.3090277777777778,
        "class_counts": {
          "ARTICULATE": 79,
          "BREAKTHROUGH": 10,
          "CONCEPTUAL": 49,
          "REPETITIVE": 41,
          "SURFACE": 109
        },
        "mean_generated_tokens": 125.59027777777777,
        "mean_output_rv": 0.6519611218558613,
        "n": 288,
        "std_output_rv": 0.1452886464142438
      },
      "total_alpha": 2.125
    },
    "anchor_early_mlp_0p125_bridge_3": {
      "alphas": {
        "bridge": 3.0,
        "early_mlp": 0.125
      },
      "by_prompt_mode": {
        "baseline": {
          "bt_art_rate": 0.22916666666666666,
          "class_counts": {
            "ARTICULATE": 30,
            "BREAKTHROUGH": 3,
            "CONCEPTUAL": 30,
            "REPETITIVE": 9,
            "SURFACE": 72
          },
          "mean_generated_tokens": 126.39583333333333,
          "mean_output_rv": 0.667644683453483,
          "n": 144,
          "std_output_rv": 0.15149785456691076
        },
        "recursive": {
          "bt_art_rate": 0.5416666666666666,
          "class_counts": {
            "ARTICULATE": 66,
            "BREAKTHROUGH": 12,
            "CONCEPTUAL": 24,
            "REPETITIVE": 20,
            "SURFACE": 22
          },
          "mean_generated_tokens": 124.99305555555556,
          "mean_output_rv": 0.6170851711847338,
          "n": 144,
          "std_output_rv": 0.12782875456140913
        }
      },
      "overall": {
        "bt_art_rate": 0.3854166666666667,
        "class_counts": {
          "ARTICULATE": 96,
          "BREAKTHROUGH": 15,
          "CONCEPTUAL": 54,
          "REPETITIVE": 29,
          "SURFACE": 94
        },
        "mean_generated_tokens": 125.69444444444444,
        "mean_output_rv": 0.6423649273191084,
        "n": 288,
        "std_output_rv": 0.14219259966762876
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
          "bt_art_rate": 0.08333333333333333,
          "class_counts": {
            "ARTICULATE": 11,
            "BREAKTHROUGH": 1,
            "CONCEPTUAL": 21,
            "REPETITIVE": 15,
            "SURFACE": 96
          },
          "mean_generated_tokens": 126.94444444444444,
          "mean_output_rv": 0.688438183650871,
          "n": 144,
          "std_output_rv": 0.1508083592475033
        },
        "recursive": {
          "bt_art_rate": 0.3055555555555556,
          "class_counts": {
            "ARTICULATE": 36,
            "BREAKTHROUGH": 8,
            "CONCEPTUAL": 21,
            "REPETITIVE": 41,
            "SURFACE": 38
          },
          "mean_generated_tokens": 125.47222222222223,
          "mean_output_rv": 0.656723235527132,
          "n": 144,
          "std_output_rv": 0.15146051529897686
        }
      },
      "overall": {
        "bt_art_rate": 0.19444444444444445,
        "class_counts": {
          "ARTICULATE": 47,
          "BREAKTHROUGH": 9,
          "CONCEPTUAL": 42,
          "REPETITIVE": 56,
          "SURFACE": 134
        },
        "mean_generated_tokens": 126.20833333333333,
        "mean_output_rv": 0.6725807095890014,
        "n": 288,
        "std_output_rv": 0.1517052148601212
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
          "bt_art_rate": 0.0763888888888889,
          "class_counts": {
            "ARTICULATE": 10,
            "BREAKTHROUGH": 1,
            "CONCEPTUAL": 13,
            "REPETITIVE": 6,
            "SURFACE": 114
          },
          "mean_generated_tokens": 122.79861111111111,
          "mean_output_rv": 0.6443997386298312,
          "n": 144,
          "std_output_rv": 0.16379464879026828
        },
        "recursive": {
          "bt_art_rate": 0.5555555555555556,
          "class_counts": {
            "ARTICULATE": 74,
            "BREAKTHROUGH": 6,
            "CONCEPTUAL": 29,
            "REPETITIVE": 17,
            "SURFACE": 18
          },
          "mean_generated_tokens": 124.34722222222223,
          "mean_output_rv": 0.6188870927883419,
          "n": 144,
          "std_output_rv": 0.108023007619091
        }
      },
      "overall": {
        "bt_art_rate": 0.3159722222222222,
        "class_counts": {
          "ARTICULATE": 84,
          "BREAKTHROUGH": 7,
          "CONCEPTUAL": 42,
          "REPETITIVE": 23,
          "SURFACE": 132
        },
        "mean_generated_tokens": 123.57291666666667,
        "mean_output_rv": 0.6316434157090866,
        "n": 288,
        "std_output_rv": 0.13908648475263385
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
          "bt_art_rate": 0.04861111111111111,
          "class_counts": {
            "ARTICULATE": 7,
            "CONCEPTUAL": 3,
            "REPETITIVE": 7,
            "SURFACE": 127
          },
          "mean_generated_tokens": 124.79861111111111,
          "mean_output_rv": 0.6328489696164203,
          "n": 144,
          "std_output_rv": 0.13858928124933964
        },
        "recursive": {
          "bt_art_rate": 0.3055555555555556,
          "class_counts": {
            "ARTICULATE": 36,
            "BREAKTHROUGH": 8,
            "CONCEPTUAL": 21,
            "REPETITIVE": 41,
            "SURFACE": 38
          },
          "mean_generated_tokens": 125.47222222222223,
          "mean_output_rv": 0.656723235527132,
          "n": 144,
          "std_output_rv": 0.15146051529897686
        }
      },
      "overall": {
        "bt_art_rate": 0.17708333333333334,
        "class_counts": {
          "ARTICULATE": 43,
          "BREAKTHROUGH": 8,
          "CONCEPTUAL": 24,
          "REPETITIVE": 48,
          "SURFACE": 165
        },
        "mean_generated_tokens": 125.13541666666667,
        "mean_output_rv": 0.6447861025717762,
        "n": 288,
        "std_output_rv": 0.14540702395953894
      },
      "total_alpha": 0.0
    },
    "early_mlp_0p125_bridge_2": {
      "alphas": {
        "bridge": 2.0,
        "early_mlp": 0.125
      },
      "by_prompt_mode": {
        "baseline": {
          "bt_art_rate": 0.0625,
          "class_counts": {
            "ARTICULATE": 9,
            "CONCEPTUAL": 9,
            "REPETITIVE": 6,
            "SURFACE": 120
          },
          "mean_generated_tokens": 124.50694444444444,
          "mean_output_rv": 0.625919779007636,
          "n": 144,
          "std_output_rv": 0.14144364220398453
        },
        "recursive": {
          "bt_art_rate": 0.5138888888888888,
          "class_counts": {
            "ARTICULATE": 67,
            "BREAKTHROUGH": 7,
            "CONCEPTUAL": 25,
            "REPETITIVE": 25,
            "SURFACE": 20
          },
          "mean_generated_tokens": 124.24305555555556,
          "mean_output_rv": 0.6124480127233038,
          "n": 144,
          "std_output_rv": 0.12975432565428718
        }
      },
      "overall": {
        "bt_art_rate": 0.2881944444444444,
        "class_counts": {
          "ARTICULATE": 76,
          "BREAKTHROUGH": 7,
          "CONCEPTUAL": 34,
          "REPETITIVE": 31,
          "SURFACE": 140
        },
        "mean_generated_tokens": 124.375,
        "mean_output_rv": 0.6191838958654698,
        "n": 288,
        "std_output_rv": 0.1356561431151328
      },
      "total_alpha": 2.125
    },
    "early_mlp_0p125_bridge_3": {
      "alphas": {
        "bridge": 3.0,
        "early_mlp": 0.125
      },
      "by_prompt_mode": {
        "baseline": {
          "bt_art_rate": 0.1111111111111111,
          "class_counts": {
            "ARTICULATE": 16,
            "CONCEPTUAL": 11,
            "REPETITIVE": 5,
            "SURFACE": 112
          },
          "mean_generated_tokens": 122.94444444444444,
          "mean_output_rv": 0.6280702932749849,
          "n": 144,
          "std_output_rv": 0.1456424010783752
        },
        "recursive": {
          "bt_art_rate": 0.5416666666666666,
          "class_counts": {
            "ARTICULATE": 66,
            "BREAKTHROUGH": 12,
            "CONCEPTUAL": 24,
            "REPETITIVE": 20,
            "SURFACE": 22
          },
          "mean_generated_tokens": 124.99305555555556,
          "mean_output_rv": 0.6170851711847338,
          "n": 144,
          "std_output_rv": 0.12782875456140913
        }
      },
      "overall": {
        "bt_art_rate": 0.3263888888888889,
        "class_counts": {
          "ARTICULATE": 82,
          "BREAKTHROUGH": 12,
          "CONCEPTUAL": 35,
          "REPETITIVE": 25,
          "SURFACE": 134
        },
        "mean_generated_tokens": 123.96875,
        "mean_output_rv": 0.6225777322298592,
        "n": 288,
        "std_output_rv": 0.13689704765710434
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
        "p": 0.0015055316485020408,
        "r": 0.26211267113514397
      },
      "alpha_vs_output_rv": {
        "p": 0.468006649737747,
        "r": -0.060952143282801415
      },
      "bt_art_rate_by_condition": {
        "anchor_bridge_3": 0.1527777777777778,
        "anchor_early_mlp_0p125_bridge_2": 0.10416666666666667,
        "anchor_early_mlp_0p125_bridge_3": 0.22916666666666666,
        "anchor_only": 0.08333333333333333,
        "bridge_only_3": 0.0763888888888889,
        "control": 0.04861111111111111,
        "early_mlp_0p125_bridge_2": 0.0625,
        "early_mlp_0p125_bridge_3": 0.1111111111111111
      },
      "mean_output_rv_by_condition": {
        "anchor_bridge_3": 0.6567736331687654,
        "anchor_early_mlp_0p125_bridge_2": 0.6914742309884185,
        "anchor_early_mlp_0p125_bridge_3": 0.667644683453483,
        "anchor_only": 0.688438183650871,
        "bridge_only_3": 0.6443997386298312,
        "control": 0.6328489696164205,
        "early_mlp_0p125_bridge_2": 0.625919779007636,
        "early_mlp_0p125_bridge_3": 0.6280702932749849
      }
    },
    "overall": {
      "alpha_vs_bt_art": {
        "p": 1.9580537336312265e-05,
        "r": 0.24870714748734307
      },
      "alpha_vs_output_rv": {
        "p": 0.0077508983228581855,
        "r": -0.15661474193462718
      },
      "bt_art_rate_by_condition": {
        "anchor_bridge_3": 0.3541666666666667,
        "anchor_early_mlp_0p125_bridge_2": 0.3090277777777778,
        "anchor_early_mlp_0p125_bridge_3": 0.3854166666666667,
        "anchor_only": 0.19444444444444445,
        "bridge_only_3": 0.3159722222222222,
        "control": 0.17708333333333334,
        "early_mlp_0p125_bridge_2": 0.2881944444444444,
        "early_mlp_0p125_bridge_3": 0.3263888888888889
      },
      "mean_output_rv_by_condition": {
        "anchor_bridge_3": 0.6378303629785537,
        "anchor_early_mlp_0p125_bridge_2": 0.6519611218558612,
        "anchor_early_mlp_0p125_bridge_3": 0.6423649273191085,
        "anchor_only": 0.6725807095890014,
        "bridge_only_3": 0.6316434157090866,
        "control": 0.6447861025717762,
        "early_mlp_0p125_bridge_2": 0.61918389586547,
        "early_mlp_0p125_bridge_3": 0.6225777322298593
      }
    },
    "recursive": {
      "alpha_vs_bt_art": {
        "p": 1.351784935427515e-07,
        "r": 0.42226342977881404
      },
      "alpha_vs_output_rv": {
        "p": 0.00010918391850830934,
        "r": -0.3168484621803727
      },
      "bt_art_rate_by_condition": {
        "anchor_bridge_3": 0.5555555555555556,
        "anchor_early_mlp_0p125_bridge_2": 0.5138888888888888,
        "anchor_early_mlp_0p125_bridge_3": 0.5416666666666666,
        "anchor_only": 0.3055555555555556,
        "bridge_only_3": 0.5555555555555556,
        "control": 0.3055555555555556,
        "early_mlp_0p125_bridge_2": 0.5138888888888888,
        "early_mlp_0p125_bridge_3": 0.5416666666666666
      },
      "mean_output_rv_by_condition": {
        "anchor_bridge_3": 0.6188870927883419,
        "anchor_early_mlp_0p125_bridge_2": 0.612448012723304,
        "anchor_early_mlp_0p125_bridge_3": 0.6170851711847337,
        "anchor_only": 0.656723235527132,
        "bridge_only_3": 0.6188870927883419,
        "control": 0.656723235527132,
        "early_mlp_0p125_bridge_2": 0.612448012723304,
        "early_mlp_0p125_bridge_3": 0.6170851711847337
      }
    }
  },
  "early_layer": 5,
  "effects_by_prompt_mode": {
    "baseline": {
      "anchor_bridge_3": {
        "alpha": 3.0,
        "bt_art_cohens_h": 0.35853718359371434,
        "bt_art_exact_sign_p": 0.03857421875,
        "bt_art_prompt_losses": 2,
        "bt_art_prompt_wins": 10,
        "bt_art_rate_control": 0.04861111111111111,
        "bt_art_rate_delta": 0.10416666666666667,
        "bt_art_rate_delta_ci_95": [
          0.034722222222222224,
          0.18055555555555555
        ],
        "bt_art_rate_treated": 0.1527777777777778,
        "n_prompt_pairs": 18,
        "rv_cohens_dz": 0.296541641321015,
        "rv_delta_ci_95": [
          -0.01203643714227827,
          0.05977150349786623
        ],
        "rv_delta_mean": 0.023924663552345054,
        "rv_p_value": 0.22535497841414134
      },
      "anchor_early_mlp_0p125_bridge_2": {
        "alpha": 2.125,
        "bt_art_cohens_h": 0.21265253145236485,
        "bt_art_exact_sign_p": 0.2890625,
        "bt_art_prompt_losses": 2,
        "bt_art_prompt_wins": 6,
        "bt_art_rate_control": 0.04861111111111111,
        "bt_art_rate_delta": 0.05555555555555555,
        "bt_art_rate_delta_ci_95": [
          0.0,
          0.11805555555555555
        ],
        "bt_art_rate_treated": 0.10416666666666667,
        "n_prompt_pairs": 18,
        "rv_cohens_dz": 0.8758928470658799,
        "rv_delta_ci_95": [
          0.030138545455164326,
          0.09012272365732775
        ],
        "rv_delta_mean": 0.058625261371998354,
        "rv_p_value": 0.0017167607594788642
      },
      "anchor_early_mlp_0p125_bridge_3": {
        "alpha": 3.125,
        "bt_art_cohens_h": 0.553766132908925,
        "bt_art_exact_sign_p": 0.000518798828125,
        "bt_art_prompt_losses": 1,
        "bt_art_prompt_wins": 15,
        "bt_art_rate_control": 0.04861111111111111,
        "bt_art_rate_delta": 0.18055555555555555,
        "bt_art_rate_delta_ci_95": [
          0.11805555555555555,
          0.24305555555555555
        ],
        "bt_art_rate_treated": 0.22916666666666666,
        "n_prompt_pairs": 18,
        "rv_cohens_dz": 0.46955457639279863,
        "rv_delta_ci_95": [
          0.003491965209643442,
          0.06875137244615166
        ],
        "rv_delta_mean": 0.034795713837062675,
        "rv_p_value": 0.06266616477749196
      },
      "anchor_only": {
        "alpha": 0.0,
        "bt_art_cohens_h": 0.14107392166399957,
        "bt_art_exact_sign_p": 0.5078125,
        "bt_art_prompt_losses": 3,
        "bt_art_prompt_wins": 6,
        "bt_art_rate_control": 0.04861111111111111,
        "bt_art_rate_delta": 0.034722222222222224,
        "bt_art_rate_delta_ci_95": [
          -0.013888888888888888,
          0.08333333333333333
        ],
        "bt_art_rate_treated": 0.08333333333333333,
        "n_prompt_pairs": 18,
        "rv_cohens_dz": 0.6858596462472283,
        "rv_delta_ci_95": [
          0.020605535938015892,
          0.09184514582936275
        ],
        "rv_delta_mean": 0.05558921403445069,
        "rv_p_value": 0.009756572651868606
      },
      "bridge_only_3": {
        "alpha": 3.0,
        "bt_art_cohens_h": 0.11545029089867387,
        "bt_art_exact_sign_p": 0.7265625,
        "bt_art_prompt_losses": 3,
        "bt_art_prompt_wins": 5,
        "bt_art_rate_control": 0.04861111111111111,
        "bt_art_rate_delta": 0.027777777777777776,
        "bt_art_rate_delta_ci_95": [
          -0.020833333333333332,
          0.0763888888888889
        ],
        "bt_art_rate_treated": 0.0763888888888889,
        "n_prompt_pairs": 18,
        "rv_cohens_dz": 0.21949661581960575,
        "rv_delta_ci_95": [
          -0.011941101572551896,
          0.03472436201579572
        ],
        "rv_delta_mean": 0.011550769013410876,
        "rv_p_value": 0.364763938124184
      },
      "early_mlp_0p125_bridge_2": {
        "alpha": 2.125,
        "bt_art_cohens_h": 0.060748888491005903,
        "bt_art_exact_sign_p": 1.0,
        "bt_art_prompt_losses": 4,
        "bt_art_prompt_wins": 5,
        "bt_art_rate_control": 0.04861111111111111,
        "bt_art_rate_delta": 0.013888888888888888,
        "bt_art_rate_delta_ci_95": [
          -0.027777777777777776,
          0.0625
        ],
        "bt_art_rate_treated": 0.0625,
        "n_prompt_pairs": 18,
        "rv_cohens_dz": -0.12645269460795447,
        "rv_delta_ci_95": [
          -0.031092139468580273,
          0.017517679441474143
        ],
        "rv_delta_mean": -0.006929190608784312,
        "rv_p_value": 0.5985688707141691
      },
      "early_mlp_0p125_bridge_3": {
        "alpha": 3.125,
        "bt_art_cohens_h": 0.23506219711509246,
        "bt_art_exact_sign_p": 0.14599609375,
        "bt_art_prompt_losses": 3,
        "bt_art_prompt_wins": 9,
        "bt_art_rate_control": 0.04861111111111111,
        "bt_art_rate_delta": 0.0625,
        "bt_art_rate_delta_ci_95": [
          0.006944444444444444,
          0.125
        ],
        "bt_art_rate_treated": 0.1111111111111111,
        "n_prompt_pairs": 18,
        "rv_cohens_dz": -0.11064149607027472,
        "rv_delta_ci_95": [
          -0.02331152501145653,
          0.015222638099515306
        ],
        "rv_delta_mean": -0.0047786763414354545,
        "rv_p_value": 0.6447394662145377
      }
    },
    "overall": {
      "anchor_bridge_3": {
        "alpha": 3.0,
        "bt_art_cohens_h": 0.40614547224623465,
        "bt_art_exact_sign_p": 0.0005461126565933228,
        "bt_art_prompt_losses": 5,
        "bt_art_prompt_wins": 24,
        "bt_art_rate_control": 0.17708333333333334,
        "bt_art_rate_delta": 0.17708333333333334,
        "bt_art_rate_delta_ci_95": [
          0.1111111111111111,
          0.24305555555555555
        ],
        "bt_art_rate_treated": 0.3541666666666667,
        "n_prompt_pairs": 36,
        "rv_cohens_dz": -0.08748189939739527,
        "rv_delta_ci_95": [
          -0.03213803504954702,
          0.017952477577741825
        ],
        "rv_delta_mean": -0.006955739593222555,
        "rv_p_value": 0.6029679198118557
      },
      "anchor_early_mlp_0p125_bridge_2": {
        "alpha": 2.125,
        "bt_art_cohens_h": 0.3102149362901293,
        "bt_art_exact_sign_p": 0.01690053939819336,
        "bt_art_prompt_losses": 5,
        "bt_art_prompt_wins": 17,
        "bt_art_rate_control": 0.17708333333333334,
        "bt_art_rate_delta": 0.13194444444444445,
        "bt_art_rate_delta_ci_95": [
          0.059027777777777776,
          0.20833333333333334
        ],
        "bt_art_rate_treated": 0.3090277777777778,
        "n_prompt_pairs": 36,
        "rv_cohens_dz": 0.08215476115784648,
        "rv_delta_ci_95": [
          -0.02097707338592666,
          0.03642358510324324
        ],
        "rv_delta_mean": 0.007175019284085125,
        "rv_p_value": 0.6251415760230732
      },
      "anchor_early_mlp_0p125_bridge_3": {
        "alpha": 3.125,
        "bt_art_cohens_h": 0.470892796147817,
        "bt_art_exact_sign_p": 1.0928604751825333e-05,
        "bt_art_prompt_losses": 4,
        "bt_art_prompt_wins": 29,
        "bt_art_rate_control": 0.17708333333333334,
        "bt_art_rate_delta": 0.20833333333333334,
        "bt_art_rate_delta_ci_95": [
          0.1423611111111111,
          0.2777777777777778
        ],
        "bt_art_rate_treated": 0.3854166666666667,
        "n_prompt_pairs": 36,
        "rv_cohens_dz": -0.029336374419378135,
        "rv_delta_ci_95": [
          -0.030193689312326178,
          0.02361497727135937
        ],
        "rv_delta_mean": -0.0024211752526677732,
        "rv_p_value": 0.8612941053964864
      },
      "anchor_only": {
        "alpha": 0.0,
        "bt_art_cohens_h": 0.04465071922352215,
        "bt_art_exact_sign_p": 0.5078125,
        "bt_art_prompt_losses": 3,
        "bt_art_prompt_wins": 6,
        "bt_art_rate_control": 0.17708333333333334,
        "bt_art_rate_delta": 0.017361111111111112,
        "bt_art_rate_delta_ci_95": [
          -0.006944444444444444,
          0.04513888888888889
        ],
        "bt_art_rate_treated": 0.19444444444444445,
        "n_prompt_pairs": 36,
        "rv_cohens_dz": 0.4402781609943269,
        "rv_delta_ci_95": [
          0.00864483619870201,
          0.049527798446245885
        ],
        "rv_delta_mean": 0.027794607017225345,
        "rv_p_value": 0.012249508081765387
      },
      "bridge_only_3": {
        "alpha": 3.0,
        "bt_art_cohens_h": 0.3251973643808056,
        "bt_art_exact_sign_p": 0.01463329792022705,
        "bt_art_prompt_losses": 6,
        "bt_art_prompt_wins": 19,
        "bt_art_rate_control": 0.17708333333333334,
        "bt_art_rate_delta": 0.1388888888888889,
        "bt_art_rate_delta_ci_95": [
          0.07291666666666667,
          0.20833333333333334
        ],
        "bt_art_rate_treated": 0.3159722222222222,
        "n_prompt_pairs": 36,
        "rv_cohens_dz": -0.20396983863929458,
        "rv_delta_ci_95": [
          -0.03383331063044104,
          0.007056585554829347
        ],
        "rv_delta_mean": -0.013142686862689644,
        "rv_p_value": 0.22919552092270676
      },
      "early_mlp_0p125_bridge_2": {
        "alpha": 2.125,
        "bt_art_cohens_h": 0.2646861943654836,
        "bt_art_exact_sign_p": 0.0931396484375,
        "bt_art_prompt_losses": 7,
        "bt_art_prompt_wins": 16,
        "bt_art_rate_control": 0.17708333333333334,
        "bt_art_rate_delta": 0.1111111111111111,
        "bt_art_rate_delta_ci_95": [
          0.041666666666666664,
          0.1875
        ],
        "bt_art_rate_treated": 0.2881944444444444,
        "n_prompt_pairs": 36,
        "rv_cohens_dz": -0.37972545010844905,
        "rv_delta_ci_95": [
          -0.046757864188426264,
          -0.003821755966823565
        ],
        "rv_delta_mean": -0.02560220670630621,
        "rv_p_value": 0.028919785893648322
      },
      "early_mlp_0p125_bridge_3": {
        "alpha": 3.125,
        "bt_art_cohens_h": 0.3475068715782529,
        "bt_art_exact_sign_p": 0.002315700054168701,
        "bt_art_prompt_losses": 6,
        "bt_art_prompt_wins": 23,
        "bt_art_rate_control": 0.17708333333333334,
        "bt_art_rate_delta": 0.14930555555555555,
        "bt_art_rate_delta_ci_95": [
          0.0798611111111111,
          0.2222222222222222
        ],
        "bt_art_rate_treated": 0.3263888888888889,
        "n_prompt_pairs": 36,
        "rv_cohens_dz": -0.3538945056373919,
        "rv_delta_ci_95": [
          -0.043569721773822026,
          -0.002603314014258103
        ],
        "rv_delta_mean": -0.02220837034191684,
        "rv_p_value": 0.040867460235302966
      }
    },
    "recursive": {
      "anchor_bridge_3": {
        "alpha": 3.0,
        "bt_art_cohens_h": 0.5107662542215583,
        "bt_art_exact_sign_p": 0.012725830078125,
        "bt_art_prompt_losses": 3,
        "bt_art_prompt_wins": 14,
        "bt_art_rate_control": 0.3055555555555556,
        "bt_art_rate_delta": 0.25,
        "bt_art_rate_delta_ci_95": [
          0.1388888888888889,
          0.3541666666666667
        ],
        "bt_art_rate_treated": 0.5555555555555556,
        "n_prompt_pairs": 18,
        "rv_cohens_dz": -0.5648269427965438,
        "rv_delta_ci_95": [
          -0.06925729634400779,
          -0.008491401643565065
        ],
        "rv_delta_mean": -0.037836142738790164,
        "rv_p_value": 0.02833441637948349
      },
      "anchor_early_mlp_0p125_bridge_2": {
        "alpha": 2.125,
        "bt_art_cohens_h": 0.42720659114438964,
        "bt_art_exact_sign_p": 0.057373046875,
        "bt_art_prompt_losses": 3,
        "bt_art_prompt_wins": 11,
        "bt_art_rate_control": 0.3055555555555556,
        "bt_art_rate_delta": 0.20833333333333334,
        "bt_art_rate_delta_ci_95": [
          0.0763888888888889,
          0.3333333333333333
        ],
        "bt_art_rate_treated": 0.5138888888888888,
        "n_prompt_pairs": 18,
        "rv_cohens_dz": -0.5907015402448139,
        "rv_delta_ci_95": [
          -0.07896069044170922,
          -0.012868324652932025
        ],
        "rv_delta_mean": -0.044275222803828104,
        "rv_p_value": 0.022661398821076525
      },
      "anchor_early_mlp_0p125_bridge_3": {
        "alpha": 3.125,
        "bt_art_cohens_h": 0.48285532649120944,
        "bt_art_exact_sign_p": 0.012725830078125,
        "bt_art_prompt_losses": 3,
        "bt_art_prompt_wins": 14,
        "bt_art_rate_control": 0.3055555555555556,
        "bt_art_rate_delta": 0.2361111111111111,
        "bt_art_rate_delta_ci_95": [
          0.125,
          0.3541666666666667
        ],
        "bt_art_rate_treated": 0.5416666666666666,
        "n_prompt_pairs": 18,
        "rv_cohens_dz": -0.5297296908381228,
        "rv_delta_ci_95": [
          -0.07547545832658728,
          -0.006360229227019017
        ],
        "rv_delta_mean": -0.03963806434239822,
        "rv_p_value": 0.03817524708981222
      },
      "anchor_only": {
        "alpha": 0.0,
        "bt_art_cohens_h": 0.0,
        "bt_art_exact_sign_p": null,
        "bt_art_prompt_losses": 0,
        "bt_art_prompt_wins": 0,
        "bt_art_rate_control": 0.3055555555555556,
        "bt_art_rate_delta": 0.0,
        "bt_art_rate_delta_ci_95": [
          0.0,
          0.0
        ],
        "bt_art_rate_treated": 0.3055555555555556,
        "n_prompt_pairs": 18,
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
        "bt_art_cohens_h": 0.5107662542215583,
        "bt_art_exact_sign_p": 0.012725830078125,
        "bt_art_prompt_losses": 3,
        "bt_art_prompt_wins": 14,
        "bt_art_rate_control": 0.3055555555555556,
        "bt_art_rate_delta": 0.25,
        "bt_art_rate_delta_ci_95": [
          0.1388888888888889,
          0.3541666666666667
        ],
        "bt_art_rate_treated": 0.5555555555555556,
        "n_prompt_pairs": 18,
        "rv_cohens_dz": -0.5648269427965438,
        "rv_delta_ci_95": [
          -0.06925729634400779,
          -0.008491401643565065
        ],
        "rv_delta_mean": -0.037836142738790164,
        "rv_p_value": 0.02833441637948349
      },
      "early_mlp_0p125_bridge_2": {
        "alpha": 2.125,
        "bt_art_cohens_h": 0.42720659114438964,
        "bt_art_exact_sign_p": 0.057373046875,
        "bt_art_prompt_losses": 3,
        "bt_art_prompt_wins": 11,
        "bt_art_rate_control": 0.3055555555555556,
        "bt_art_rate_delta": 0.20833333333333334,
        "bt_art_rate_delta_ci_95": [
          0.0763888888888889,
          0.3333333333333333
        ],
        "bt_art_rate_treated": 0.5138888888888888,
        "n_prompt_pairs": 18,
        "rv_cohens_dz": -0.5907015402448139,
        "rv_delta_ci_95": [
          -0.07896069044170922,
          -0.012868324652932025
        ],
        "rv_delta_mean": -0.044275222803828104,
        "rv_p_value": 0.022661398821076525
      },
      "early_mlp_0p125_bridge_3": {
        "alpha": 3.125,
        "bt_art_cohens_h": 0.48285532649120944,
        "bt_art_exact_sign_p": 0.012725830078125,
        "bt_art_prompt_losses": 3,
        "bt_art_prompt_wins": 14,
        "bt_art_rate_control": 0.3055555555555556,
        "bt_art_rate_delta": 0.2361111111111111,
        "bt_art_rate_delta_ci_95": [
          0.125,
          0.3541666666666667
        ],
        "bt_art_rate_treated": 0.5416666666666666,
        "n_prompt_pairs": 18,
        "rv_cohens_dz": -0.5297296908381228,
        "rv_delta_ci_95": [
          -0.07547545832658728,
          -0.006360229227019017
        ],
        "rv_delta_mean": -0.03963806434239822,
        "rv_p_value": 0.03817524708981222
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
      "L3_deeper": 6,
      "L4_full": 6,
      "L5_refined": 6,
      "baseline_creative": 6,
      "baseline_factual": 6,
      "baseline_math": 6
    },
    "by_mode": {
      "baseline": 18,
      "recursive": 18
    },
    "total": 36
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
        "bridge": 2.0,
        "early_mlp": 0.125
      },
      "name": "early_mlp_0p125_bridge_2"
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
        "bridge": 2.0,
        "early_mlp": 0.125
      },
      "name": "anchor_early_mlp_0p125_bridge_2",
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
  "n_pairs": 36,
  "n_total": 36,
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
  "timestamp": "20260314_041952",
  "top_p": 0.95,
  "verdict": "multisite_sufficient"
}
```
