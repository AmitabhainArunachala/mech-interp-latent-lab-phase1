# Run report: causal_state_benchmark_v4_multisite

- **run_dir**: `results/phase1_mechanism/runs/20260312_133738_causal_state_benchmark_v4_multisite_mistral_multisite_l4_mlp_micro4_l25_bridge`
- **prompt_bank_version**: `2ac959a313614329`

## Summary (machine-readable)

```json
{
  "artifacts": {
    "blind_key_json": "results/phase1_mechanism/runs/20260312_133738_causal_state_benchmark_v4_multisite_mistral_multisite_l4_mlp_micro4_l25_bridge/blind_key.json",
    "blind_ratings_csv": "results/phase1_mechanism/runs/20260312_133738_causal_state_benchmark_v4_multisite_mistral_multisite_l4_mlp_micro4_l25_bridge/blind_ratings.csv",
    "config": "results/phase1_mechanism/runs/20260312_133738_causal_state_benchmark_v4_multisite_mistral_multisite_l4_mlp_micro4_l25_bridge/config.json",
    "manifest": "results/phase1_mechanism/runs/20260312_133738_causal_state_benchmark_v4_multisite_mistral_multisite_l4_mlp_micro4_l25_bridge/manifest.json",
    "records_jsonl": "results/phase1_mechanism/runs/20260312_133738_causal_state_benchmark_v4_multisite_mistral_multisite_l4_mlp_micro4_l25_bridge/benchmark_records.jsonl",
    "report": "results/phase1_mechanism/runs/20260312_133738_causal_state_benchmark_v4_multisite_mistral_multisite_l4_mlp_micro4_l25_bridge/report.md",
    "state_directions_pt": "results/phase1_mechanism/runs/20260312_133738_causal_state_benchmark_v4_multisite_mistral_multisite_l4_mlp_micro4_l25_bridge/state_directions.pt",
    "summary": "results/phase1_mechanism/runs/20260312_133738_causal_state_benchmark_v4_multisite_mistral_multisite_l4_mlp_micro4_l25_bridge/summary.json"
  },
  "bootstrap_resamples": 3000,
  "by_condition": {
    "bridge_only_2": {
      "alphas": {
        "bridge": 2.0,
        "early_mlp": 0.0
      },
      "by_prompt_mode": {
        "baseline": {
          "bt_art_rate": 0.08333333333333333,
          "class_counts": {
            "ARTICULATE": 3,
            "CONCEPTUAL": 1,
            "REPETITIVE": 1,
            "SURFACE": 31
          },
          "mean_generated_tokens": 127.27777777777777,
          "mean_output_rv": 0.638012004826476,
          "n": 36,
          "std_output_rv": 0.16171009251773813
        },
        "recursive": {
          "bt_art_rate": 0.16666666666666666,
          "class_counts": {
            "ARTICULATE": 6,
            "CONCEPTUAL": 9,
            "REPETITIVE": 12,
            "SURFACE": 9
          },
          "mean_generated_tokens": 124.58333333333333,
          "mean_output_rv": 0.6713644001897969,
          "n": 36,
          "std_output_rv": 0.12554687220435853
        }
      },
      "overall": {
        "bt_art_rate": 0.125,
        "class_counts": {
          "ARTICULATE": 9,
          "CONCEPTUAL": 10,
          "REPETITIVE": 13,
          "SURFACE": 40
        },
        "mean_generated_tokens": 125.93055555555556,
        "mean_output_rv": 0.6546882025081364,
        "n": 72,
        "std_output_rv": 0.14471676421379237
      },
      "total_alpha": 2.0
    },
    "bridge_only_3": {
      "alphas": {
        "bridge": 3.0,
        "early_mlp": 0.0
      },
      "by_prompt_mode": {
        "baseline": {
          "bt_art_rate": 0.1388888888888889,
          "class_counts": {
            "ARTICULATE": 5,
            "CONCEPTUAL": 4,
            "SURFACE": 27
          },
          "mean_generated_tokens": 126.11111111111111,
          "mean_output_rv": 0.6435709751288297,
          "n": 36,
          "std_output_rv": 0.1560145613764485
        },
        "recursive": {
          "bt_art_rate": 0.4444444444444444,
          "class_counts": {
            "ARTICULATE": 16,
            "CONCEPTUAL": 9,
            "REPETITIVE": 7,
            "SURFACE": 4
          },
          "mean_generated_tokens": 122.41666666666667,
          "mean_output_rv": 0.6430804655506568,
          "n": 36,
          "std_output_rv": 0.11762041303023518
        }
      },
      "overall": {
        "bt_art_rate": 0.2916666666666667,
        "class_counts": {
          "ARTICULATE": 21,
          "CONCEPTUAL": 13,
          "REPETITIVE": 7,
          "SURFACE": 31
        },
        "mean_generated_tokens": 124.26388888888889,
        "mean_output_rv": 0.6433257203397432,
        "n": 72,
        "std_output_rv": 0.1371815391788385
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
          "bt_art_rate": 0.05555555555555555,
          "class_counts": {
            "ARTICULATE": 2,
            "CONCEPTUAL": 2,
            "REPETITIVE": 3,
            "SURFACE": 29
          },
          "mean_generated_tokens": 126.19444444444444,
          "mean_output_rv": 0.6310534379177899,
          "n": 36,
          "std_output_rv": 0.13076951160559863
        },
        "recursive": {
          "bt_art_rate": 0.25,
          "class_counts": {
            "ARTICULATE": 7,
            "BREAKTHROUGH": 2,
            "CONCEPTUAL": 3,
            "REPETITIVE": 13,
            "SURFACE": 11
          },
          "mean_generated_tokens": 124.55555555555556,
          "mean_output_rv": 0.6570764282822219,
          "n": 36,
          "std_output_rv": 0.1708981031731647
        }
      },
      "overall": {
        "bt_art_rate": 0.1527777777777778,
        "class_counts": {
          "ARTICULATE": 9,
          "BREAKTHROUGH": 2,
          "CONCEPTUAL": 5,
          "REPETITIVE": 16,
          "SURFACE": 40
        },
        "mean_generated_tokens": 125.375,
        "mean_output_rv": 0.6440649331000059,
        "n": 72,
        "std_output_rv": 0.15165418722371451
      },
      "total_alpha": 0.0
    },
    "early_mlp_0p03125_bridge_2": {
      "alphas": {
        "bridge": 2.0,
        "early_mlp": 0.03125
      },
      "by_prompt_mode": {
        "baseline": {
          "bt_art_rate": 0.1111111111111111,
          "class_counts": {
            "ARTICULATE": 4,
            "CONCEPTUAL": 2,
            "REPETITIVE": 1,
            "SURFACE": 29
          },
          "mean_generated_tokens": 126.30555555555556,
          "mean_output_rv": 0.6092562532841544,
          "n": 36,
          "std_output_rv": 0.13563036729986933
        },
        "recursive": {
          "bt_art_rate": 0.2222222222222222,
          "class_counts": {
            "ARTICULATE": 8,
            "CONCEPTUAL": 8,
            "REPETITIVE": 13,
            "SURFACE": 7
          },
          "mean_generated_tokens": 124.58333333333333,
          "mean_output_rv": 0.6492805811524632,
          "n": 36,
          "std_output_rv": 0.1319199216216585
        }
      },
      "overall": {
        "bt_art_rate": 0.16666666666666666,
        "class_counts": {
          "ARTICULATE": 12,
          "CONCEPTUAL": 10,
          "REPETITIVE": 14,
          "SURFACE": 36
        },
        "mean_generated_tokens": 125.44444444444444,
        "mean_output_rv": 0.629268417218309,
        "n": 72,
        "std_output_rv": 0.1343624078511801
      },
      "total_alpha": 2.03125
    },
    "early_mlp_0p03125_bridge_3": {
      "alphas": {
        "bridge": 3.0,
        "early_mlp": 0.03125
      },
      "by_prompt_mode": {
        "baseline": {
          "bt_art_rate": 0.027777777777777776,
          "class_counts": {
            "ARTICULATE": 1,
            "CONCEPTUAL": 4,
            "REPETITIVE": 1,
            "SURFACE": 30
          },
          "mean_generated_tokens": 124.97222222222223,
          "mean_output_rv": 0.6373670983030937,
          "n": 36,
          "std_output_rv": 0.15454225503869892
        },
        "recursive": {
          "bt_art_rate": 0.5277777777777778,
          "class_counts": {
            "ARTICULATE": 17,
            "BREAKTHROUGH": 2,
            "CONCEPTUAL": 5,
            "REPETITIVE": 7,
            "SURFACE": 5
          },
          "mean_generated_tokens": 122.05555555555556,
          "mean_output_rv": 0.6335586683258788,
          "n": 36,
          "std_output_rv": 0.1279164076773201
        }
      },
      "overall": {
        "bt_art_rate": 0.2777777777777778,
        "class_counts": {
          "ARTICULATE": 18,
          "BREAKTHROUGH": 2,
          "CONCEPTUAL": 9,
          "REPETITIVE": 8,
          "SURFACE": 35
        },
        "mean_generated_tokens": 123.51388888888889,
        "mean_output_rv": 0.6354628833144863,
        "n": 72,
        "std_output_rv": 0.14086594029471586
      },
      "total_alpha": 3.03125
    },
    "early_mlp_0p0625_bridge_2": {
      "alphas": {
        "bridge": 2.0,
        "early_mlp": 0.0625
      },
      "by_prompt_mode": {
        "baseline": {
          "bt_art_rate": 0.05555555555555555,
          "class_counts": {
            "ARTICULATE": 2,
            "CONCEPTUAL": 3,
            "REPETITIVE": 1,
            "SURFACE": 30
          },
          "mean_generated_tokens": 125.66666666666667,
          "mean_output_rv": 0.6223404352430534,
          "n": 36,
          "std_output_rv": 0.14846038426538058
        },
        "recursive": {
          "bt_art_rate": 0.2777777777777778,
          "class_counts": {
            "ARTICULATE": 9,
            "BREAKTHROUGH": 1,
            "CONCEPTUAL": 7,
            "REPETITIVE": 12,
            "SURFACE": 7
          },
          "mean_generated_tokens": 123.13888888888889,
          "mean_output_rv": 0.6404978938054948,
          "n": 36,
          "std_output_rv": 0.11405005350810841
        }
      },
      "overall": {
        "bt_art_rate": 0.16666666666666666,
        "class_counts": {
          "ARTICULATE": 11,
          "BREAKTHROUGH": 1,
          "CONCEPTUAL": 10,
          "REPETITIVE": 13,
          "SURFACE": 37
        },
        "mean_generated_tokens": 124.40277777777777,
        "mean_output_rv": 0.631419164524274,
        "n": 72,
        "std_output_rv": 0.13176007824060965
      },
      "total_alpha": 2.0625
    },
    "early_mlp_0p0625_bridge_3": {
      "alphas": {
        "bridge": 3.0,
        "early_mlp": 0.0625
      },
      "by_prompt_mode": {
        "baseline": {
          "bt_art_rate": 0.08333333333333333,
          "class_counts": {
            "ARTICULATE": 3,
            "CONCEPTUAL": 4,
            "SURFACE": 29
          },
          "mean_generated_tokens": 125.5,
          "mean_output_rv": 0.6366892294961068,
          "n": 36,
          "std_output_rv": 0.16277528276854608
        },
        "recursive": {
          "bt_art_rate": 0.4444444444444444,
          "class_counts": {
            "ARTICULATE": 12,
            "BREAKTHROUGH": 4,
            "CONCEPTUAL": 8,
            "REPETITIVE": 8,
            "SURFACE": 4
          },
          "mean_generated_tokens": 123.36111111111111,
          "mean_output_rv": 0.6392416633762141,
          "n": 36,
          "std_output_rv": 0.13548388255669933
        }
      },
      "overall": {
        "bt_art_rate": 0.2638888888888889,
        "class_counts": {
          "ARTICULATE": 15,
          "BREAKTHROUGH": 4,
          "CONCEPTUAL": 12,
          "REPETITIVE": 8,
          "SURFACE": 33
        },
        "mean_generated_tokens": 124.43055555555556,
        "mean_output_rv": 0.6379654464361604,
        "n": 72,
        "std_output_rv": 0.14869980618362766
      },
      "total_alpha": 3.0625
    },
    "early_mlp_0p125_bridge_2": {
      "alphas": {
        "bridge": 2.0,
        "early_mlp": 0.125
      },
      "by_prompt_mode": {
        "baseline": {
          "bt_art_rate": 0.05555555555555555,
          "class_counts": {
            "ARTICULATE": 2,
            "CONCEPTUAL": 3,
            "REPETITIVE": 1,
            "SURFACE": 30
          },
          "mean_generated_tokens": 123.0,
          "mean_output_rv": 0.6295557558707736,
          "n": 36,
          "std_output_rv": 0.15398195353059968
        },
        "recursive": {
          "bt_art_rate": 0.4166666666666667,
          "class_counts": {
            "ARTICULATE": 13,
            "BREAKTHROUGH": 2,
            "CONCEPTUAL": 5,
            "REPETITIVE": 12,
            "SURFACE": 4
          },
          "mean_generated_tokens": 125.58333333333333,
          "mean_output_rv": 0.6399667019750587,
          "n": 36,
          "std_output_rv": 0.145014960818772
        }
      },
      "overall": {
        "bt_art_rate": 0.2361111111111111,
        "class_counts": {
          "ARTICULATE": 15,
          "BREAKTHROUGH": 2,
          "CONCEPTUAL": 8,
          "REPETITIVE": 13,
          "SURFACE": 34
        },
        "mean_generated_tokens": 124.29166666666667,
        "mean_output_rv": 0.6347612289229161,
        "n": 72,
        "std_output_rv": 0.14860114452676096
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
            "ARTICULATE": 4,
            "CONCEPTUAL": 6,
            "SURFACE": 26
          },
          "mean_generated_tokens": 125.19444444444444,
          "mean_output_rv": 0.6455327694759256,
          "n": 36,
          "std_output_rv": 0.16293760413340402
        },
        "recursive": {
          "bt_art_rate": 0.4444444444444444,
          "class_counts": {
            "ARTICULATE": 14,
            "BREAKTHROUGH": 2,
            "CONCEPTUAL": 8,
            "REPETITIVE": 7,
            "SURFACE": 5
          },
          "mean_generated_tokens": 126.36111111111111,
          "mean_output_rv": 0.5989605709876884,
          "n": 36,
          "std_output_rv": 0.1335905978366337
        }
      },
      "overall": {
        "bt_art_rate": 0.2777777777777778,
        "class_counts": {
          "ARTICULATE": 18,
          "BREAKTHROUGH": 2,
          "CONCEPTUAL": 14,
          "REPETITIVE": 7,
          "SURFACE": 31
        },
        "mean_generated_tokens": 125.77777777777777,
        "mean_output_rv": 0.622246670231807,
        "n": 72,
        "std_output_rv": 0.14978248678086256
      },
      "total_alpha": 3.125
    },
    "early_mlp_0p1875_bridge_2": {
      "alphas": {
        "bridge": 2.0,
        "early_mlp": 0.1875
      },
      "by_prompt_mode": {
        "baseline": {
          "bt_art_rate": 0.08333333333333333,
          "class_counts": {
            "ARTICULATE": 3,
            "REPETITIVE": 1,
            "SURFACE": 32
          },
          "mean_generated_tokens": 123.5,
          "mean_output_rv": 0.6448843654099488,
          "n": 36,
          "std_output_rv": 0.14804086643583825
        },
        "recursive": {
          "bt_art_rate": 0.4444444444444444,
          "class_counts": {
            "ARTICULATE": 12,
            "BREAKTHROUGH": 4,
            "CONCEPTUAL": 5,
            "REPETITIVE": 11,
            "SURFACE": 4
          },
          "mean_generated_tokens": 123.77777777777777,
          "mean_output_rv": 0.622728469876005,
          "n": 36,
          "std_output_rv": 0.12444293326391838
        }
      },
      "overall": {
        "bt_art_rate": 0.2638888888888889,
        "class_counts": {
          "ARTICULATE": 15,
          "BREAKTHROUGH": 4,
          "CONCEPTUAL": 5,
          "REPETITIVE": 12,
          "SURFACE": 36
        },
        "mean_generated_tokens": 123.63888888888889,
        "mean_output_rv": 0.6338064176429768,
        "n": 72,
        "std_output_rv": 0.13624288940372958
      },
      "total_alpha": 2.1875
    },
    "early_mlp_0p1875_bridge_3": {
      "alphas": {
        "bridge": 3.0,
        "early_mlp": 0.1875
      },
      "by_prompt_mode": {
        "baseline": {
          "bt_art_rate": 0.08333333333333333,
          "class_counts": {
            "ARTICULATE": 3,
            "CONCEPTUAL": 9,
            "SURFACE": 24
          },
          "mean_generated_tokens": 123.80555555555556,
          "mean_output_rv": 0.6373126233370697,
          "n": 36,
          "std_output_rv": 0.16359546551330387
        },
        "recursive": {
          "bt_art_rate": 0.4444444444444444,
          "class_counts": {
            "ARTICULATE": 14,
            "BREAKTHROUGH": 2,
            "CONCEPTUAL": 8,
            "REPETITIVE": 6,
            "SURFACE": 6
          },
          "mean_generated_tokens": 123.16666666666667,
          "mean_output_rv": 0.6173109800144244,
          "n": 36,
          "std_output_rv": 0.11854905491081048
        }
      },
      "overall": {
        "bt_art_rate": 0.2638888888888889,
        "class_counts": {
          "ARTICULATE": 17,
          "BREAKTHROUGH": 2,
          "CONCEPTUAL": 17,
          "REPETITIVE": 6,
          "SURFACE": 30
        },
        "mean_generated_tokens": 123.48611111111111,
        "mean_output_rv": 0.6273118016757471,
        "n": 72,
        "std_output_rv": 0.1422063864043764
      },
      "total_alpha": 3.1875
    }
  },
  "control_prompt_mode": "baseline",
  "device": "cuda",
  "do_sample": true,
  "dose_response": {
    "baseline": {
      "alpha_vs_bt_art": {
        "p": 0.5332346210560268,
        "r": 0.05471153034578529
      },
      "alpha_vs_output_rv": {
        "p": 0.6552065333920213,
        "r": 0.0392245107484858
      },
      "bt_art_rate_by_condition": {
        "bridge_only_2": 0.08333333333333333,
        "bridge_only_3": 0.13888888888888887,
        "control": 0.05555555555555555,
        "early_mlp_0p03125_bridge_2": 0.1111111111111111,
        "early_mlp_0p03125_bridge_3": 0.027777777777777776,
        "early_mlp_0p0625_bridge_2": 0.05555555555555555,
        "early_mlp_0p0625_bridge_3": 0.08333333333333333,
        "early_mlp_0p125_bridge_2": 0.05555555555555555,
        "early_mlp_0p125_bridge_3": 0.1111111111111111,
        "early_mlp_0p1875_bridge_2": 0.08333333333333333,
        "early_mlp_0p1875_bridge_3": 0.08333333333333333
      },
      "mean_output_rv_by_condition": {
        "bridge_only_2": 0.6380120048264762,
        "bridge_only_3": 0.6435709751288295,
        "control": 0.6310534379177898,
        "early_mlp_0p03125_bridge_2": 0.6092562532841547,
        "early_mlp_0p03125_bridge_3": 0.6373670983030936,
        "early_mlp_0p0625_bridge_2": 0.6223404352430536,
        "early_mlp_0p0625_bridge_3": 0.6366892294961068,
        "early_mlp_0p125_bridge_2": 0.6295557558707737,
        "early_mlp_0p125_bridge_3": 0.6455327694759256,
        "early_mlp_0p1875_bridge_2": 0.6448843654099488,
        "early_mlp_0p1875_bridge_3": 0.6373126233370696
      }
    },
    "overall": {
      "alpha_vs_bt_art": {
        "p": 0.01590044958374769,
        "r": 0.14828099414931595
      },
      "alpha_vs_output_rv": {
        "p": 0.539925913675903,
        "r": -0.03788917086919041
      },
      "bt_art_rate_by_condition": {
        "bridge_only_2": 0.125,
        "bridge_only_3": 0.2916666666666667,
        "control": 0.1527777777777778,
        "early_mlp_0p03125_bridge_2": 0.16666666666666666,
        "early_mlp_0p03125_bridge_3": 0.27777777777777773,
        "early_mlp_0p0625_bridge_2": 0.16666666666666666,
        "early_mlp_0p0625_bridge_3": 0.2638888888888889,
        "early_mlp_0p125_bridge_2": 0.23611111111111108,
        "early_mlp_0p125_bridge_3": 0.27777777777777773,
        "early_mlp_0p1875_bridge_2": 0.2638888888888889,
        "early_mlp_0p1875_bridge_3": 0.26388888888888884
      },
      "mean_output_rv_by_condition": {
        "bridge_only_2": 0.6546882025081365,
        "bridge_only_3": 0.6433257203397431,
        "control": 0.6440649331000059,
        "early_mlp_0p03125_bridge_2": 0.629268417218309,
        "early_mlp_0p03125_bridge_3": 0.6354628833144862,
        "early_mlp_0p0625_bridge_2": 0.631419164524274,
        "early_mlp_0p0625_bridge_3": 0.6379654464361604,
        "early_mlp_0p125_bridge_2": 0.6347612289229161,
        "early_mlp_0p125_bridge_3": 0.6222466702318069,
        "early_mlp_0p1875_bridge_2": 0.6338064176429768,
        "early_mlp_0p1875_bridge_3": 0.627311801675747
      }
    },
    "recursive": {
      "alpha_vs_bt_art": {
        "p": 0.004065437159171599,
        "r": 0.24849464691995862
      },
      "alpha_vs_output_rv": {
        "p": 0.1270135949775527,
        "r": -0.13349442361762684
      },
      "bt_art_rate_by_condition": {
        "bridge_only_2": 0.16666666666666666,
        "bridge_only_3": 0.44444444444444436,
        "control": 0.25,
        "early_mlp_0p03125_bridge_2": 0.2222222222222222,
        "early_mlp_0p03125_bridge_3": 0.5277777777777777,
        "early_mlp_0p0625_bridge_2": 0.27777777777777773,
        "early_mlp_0p0625_bridge_3": 0.44444444444444436,
        "early_mlp_0p125_bridge_2": 0.4166666666666666,
        "early_mlp_0p125_bridge_3": 0.4444444444444444,
        "early_mlp_0p1875_bridge_2": 0.44444444444444436,
        "early_mlp_0p1875_bridge_3": 0.44444444444444436
      },
      "mean_output_rv_by_condition": {
        "bridge_only_2": 0.6713644001897968,
        "bridge_only_3": 0.6430804655506567,
        "control": 0.657076428282222,
        "early_mlp_0p03125_bridge_2": 0.6492805811524632,
        "early_mlp_0p03125_bridge_3": 0.6335586683258787,
        "early_mlp_0p0625_bridge_2": 0.6404978938054947,
        "early_mlp_0p0625_bridge_3": 0.639241663376214,
        "early_mlp_0p125_bridge_2": 0.6399667019750587,
        "early_mlp_0p125_bridge_3": 0.5989605709876883,
        "early_mlp_0p1875_bridge_2": 0.622728469876005,
        "early_mlp_0p1875_bridge_3": 0.6173109800144242
      }
    }
  },
  "early_layer": 5,
  "effects_by_prompt_mode": {
    "baseline": {
      "bridge_only_2": {
        "alpha": 2.0,
        "bt_art_cohens_h": 0.10980329379673442,
        "bt_art_exact_sign_p": 1.0,
        "bt_art_prompt_losses": 1,
        "bt_art_prompt_wins": 1,
        "bt_art_rate_control": 0.05555555555555555,
        "bt_art_rate_delta": 0.027777777777777776,
        "bt_art_rate_delta_ci_95": [
          -0.08333333333333333,
          0.16666666666666666
        ],
        "bt_art_rate_treated": 0.08333333333333333,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": 0.07721464819958937,
        "rv_delta_ci_95": [
          -0.0411516612896098,
          0.05498992659165367
        ],
        "rv_delta_mean": 0.006958566908686306,
        "rv_p_value": 0.7940488285284413
      },
      "bridge_only_3": {
        "alpha": 3.0,
        "bt_art_cohens_h": 0.28790424673741494,
        "bt_art_exact_sign_p": 0.6875,
        "bt_art_prompt_losses": 2,
        "bt_art_prompt_wins": 4,
        "bt_art_rate_control": 0.05555555555555555,
        "bt_art_rate_delta": 0.08333333333333333,
        "bt_art_rate_delta_ci_95": [
          -0.05555555555555556,
          0.25
        ],
        "bt_art_rate_treated": 0.13888888888888887,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": 0.10629680676017737,
        "rv_delta_ci_95": [
          -0.04668072593648842,
          0.07944575694847487
        ],
        "rv_delta_mean": 0.012517537211039867,
        "rv_p_value": 0.7196949103533382
      },
      "early_mlp_0p03125_bridge_2": {
        "alpha": 2.03125,
        "bt_art_cohens_h": 0.20379156924782732,
        "bt_art_exact_sign_p": 0.5,
        "bt_art_prompt_losses": 0,
        "bt_art_prompt_wins": 2,
        "bt_art_rate_control": 0.05555555555555555,
        "bt_art_rate_delta": 0.05555555555555555,
        "bt_art_rate_delta_ci_95": [
          0.0,
          0.13888888888888887
        ],
        "bt_art_rate_treated": 0.1111111111111111,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": -0.22981701941720056,
        "rv_delta_ci_95": [
          -0.07304202230132097,
          0.029209592234208177
        ],
        "rv_delta_mean": -0.02179718463363507,
        "rv_p_value": 0.4428105083337841
      },
      "early_mlp_0p03125_bridge_3": {
        "alpha": 3.03125,
        "bt_art_cohens_h": -0.1409860912210379,
        "bt_art_exact_sign_p": 1.0,
        "bt_art_prompt_losses": 2,
        "bt_art_prompt_wins": 1,
        "bt_art_rate_control": 0.05555555555555555,
        "bt_art_rate_delta": -0.027777777777777776,
        "bt_art_rate_delta_ci_95": [
          -0.1111111111111111,
          0.05555555555555555
        ],
        "bt_art_rate_treated": 0.027777777777777776,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": 0.05623853906772366,
        "rv_delta_ci_95": [
          -0.04907985417706417,
          0.07412712199482041
        ],
        "rv_delta_mean": 0.006313660385303889,
        "rv_p_value": 0.8490885150315308
      },
      "early_mlp_0p0625_bridge_2": {
        "alpha": 2.0625,
        "bt_art_cohens_h": 0.0,
        "bt_art_exact_sign_p": null,
        "bt_art_prompt_losses": 0,
        "bt_art_prompt_wins": 0,
        "bt_art_rate_control": 0.05555555555555555,
        "bt_art_rate_delta": 0.0,
        "bt_art_rate_delta_ci_95": [
          0.0,
          0.0
        ],
        "bt_art_rate_treated": 0.05555555555555555,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": -0.08144560729634662,
        "rv_delta_ci_95": [
          -0.06312305828309875,
          0.05233056042308752
        ],
        "rv_delta_mean": -0.008713002674736231,
        "rv_p_value": 0.7830769655687031
      },
      "early_mlp_0p0625_bridge_3": {
        "alpha": 3.0625,
        "bt_art_cohens_h": 0.10980329379673442,
        "bt_art_exact_sign_p": 1.0,
        "bt_art_prompt_losses": 1,
        "bt_art_prompt_wins": 1,
        "bt_art_rate_control": 0.05555555555555555,
        "bt_art_rate_delta": 0.027777777777777776,
        "bt_art_rate_delta_ci_95": [
          -0.08333333333333333,
          0.16666666666666666
        ],
        "bt_art_rate_treated": 0.08333333333333333,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": 0.04621827824152519,
        "rv_delta_ci_95": [
          -0.05497195129109282,
          0.07699369126216918
        ],
        "rv_delta_mean": 0.005635791578317062,
        "rv_p_value": 0.875700705003039
      },
      "early_mlp_0p125_bridge_2": {
        "alpha": 2.125,
        "bt_art_cohens_h": 0.0,
        "bt_art_exact_sign_p": 1.0,
        "bt_art_prompt_losses": 2,
        "bt_art_prompt_wins": 2,
        "bt_art_rate_control": 0.05555555555555555,
        "bt_art_rate_delta": 0.0,
        "bt_art_rate_delta_ci_95": [
          -0.1111111111111111,
          0.1111111111111111
        ],
        "bt_art_rate_treated": 0.05555555555555555,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": -0.019188249709821744,
        "rv_delta_ci_95": [
          -0.04279878310008768,
          0.042257610231598366
        ],
        "rv_delta_mean": -0.0014976820470160163,
        "rv_p_value": 0.9481963384951615
      },
      "early_mlp_0p125_bridge_3": {
        "alpha": 3.125,
        "bt_art_cohens_h": 0.20379156924782732,
        "bt_art_exact_sign_p": 1.0,
        "bt_art_prompt_losses": 1,
        "bt_art_prompt_wins": 2,
        "bt_art_rate_control": 0.05555555555555555,
        "bt_art_rate_delta": 0.05555555555555555,
        "bt_art_rate_delta_ci_95": [
          -0.05555555555555555,
          0.19444444444444442
        ],
        "bt_art_rate_treated": 0.1111111111111111,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": 0.19873697415492014,
        "rv_delta_ci_95": [
          -0.02426038618130887,
          0.05343955150863404
        ],
        "rv_delta_mean": 0.01447933155813593,
        "rv_p_value": 0.5054325389646435
      },
      "early_mlp_0p1875_bridge_2": {
        "alpha": 2.1875,
        "bt_art_cohens_h": 0.10980329379673442,
        "bt_art_exact_sign_p": 1.0,
        "bt_art_prompt_losses": 1,
        "bt_art_prompt_wins": 1,
        "bt_art_rate_control": 0.05555555555555555,
        "bt_art_rate_delta": 0.027777777777777776,
        "bt_art_rate_delta_ci_95": [
          -0.08333333333333333,
          0.16666666666666666
        ],
        "bt_art_rate_treated": 0.08333333333333333,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": 0.12456183747575014,
        "rv_delta_ci_95": [
          -0.044223171445321834,
          0.07852414405989405
        ],
        "rv_delta_mean": 0.01383092749215895,
        "rv_p_value": 0.6744437986641518
      },
      "early_mlp_0p1875_bridge_3": {
        "alpha": 3.1875,
        "bt_art_cohens_h": 0.10980329379673442,
        "bt_art_exact_sign_p": 1.0,
        "bt_art_prompt_losses": 1,
        "bt_art_prompt_wins": 2,
        "bt_art_rate_control": 0.05555555555555555,
        "bt_art_rate_delta": 0.027777777777777776,
        "bt_art_rate_delta_ci_95": [
          -0.05555555555555556,
          0.1111111111111111
        ],
        "bt_art_rate_treated": 0.08333333333333333,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": 0.1115406815771538,
        "rv_delta_ci_95": [
          -0.02134994475218011,
          0.038936507024060804
        ],
        "rv_delta_mean": 0.006259185419279949,
        "rv_p_value": 0.7065785311838586
      }
    },
    "overall": {
      "bridge_only_2": {
        "alpha": 2.0,
        "bt_art_cohens_h": -0.08041455757345006,
        "bt_art_exact_sign_p": 0.6875,
        "bt_art_prompt_losses": 4,
        "bt_art_prompt_wins": 2,
        "bt_art_rate_control": 0.1527777777777778,
        "bt_art_rate_delta": -0.02777777777777778,
        "bt_art_rate_delta_ci_95": [
          -0.125,
          0.06944444444444443
        ],
        "bt_art_rate_treated": 0.125,
        "n_prompt_pairs": 24,
        "rv_cohens_dz": 0.09272820623221584,
        "rv_delta_ci_95": [
          -0.03395873185997595,
          0.05677295612633069
        ],
        "rv_delta_mean": 0.010623269408130595,
        "rv_p_value": 0.6538900090798534
      },
      "bridge_only_3": {
        "alpha": 3.0,
        "bt_art_cohens_h": 0.3378720901035034,
        "bt_art_exact_sign_p": 0.076812744140625,
        "bt_art_prompt_losses": 4,
        "bt_art_prompt_wins": 12,
        "bt_art_rate_control": 0.1527777777777778,
        "bt_art_rate_delta": 0.13888888888888887,
        "bt_art_rate_delta_ci_95": [
          0.027777777777777766,
          0.25
        ],
        "bt_art_rate_treated": 0.2916666666666667,
        "n_prompt_pairs": 24,
        "rv_cohens_dz": -0.006278236071684392,
        "rv_delta_ci_95": [
          -0.04661316303815092,
          0.04346382524896796
        ],
        "rv_delta_mean": -0.0007392127602627007,
        "rv_p_value": 0.9757287149180869
      },
      "early_mlp_0p03125_bridge_2": {
        "alpha": 2.03125,
        "bt_art_cohens_h": 0.037919865181064494,
        "bt_art_exact_sign_p": 0.453125,
        "bt_art_prompt_losses": 2,
        "bt_art_prompt_wins": 5,
        "bt_art_rate_control": 0.1527777777777778,
        "bt_art_rate_delta": 0.013888888888888883,
        "bt_art_rate_delta_ci_95": [
          -0.11111111111111112,
          0.1111111111111111
        ],
        "bt_art_rate_treated": 0.16666666666666666,
        "n_prompt_pairs": 24,
        "rv_cohens_dz": -0.12156990069631303,
        "rv_delta_ci_95": [
          -0.06178816254983474,
          0.032293444687765495
        ],
        "rv_delta_mean": -0.01479651588169692,
        "rv_p_value": 0.5572791563331607
      },
      "early_mlp_0p03125_bridge_3": {
        "alpha": 3.03125,
        "bt_art_cohens_h": 0.3070935297267082,
        "bt_art_exact_sign_p": 0.11846923828124999,
        "bt_art_prompt_losses": 4,
        "bt_art_prompt_wins": 11,
        "bt_art_rate_control": 0.1527777777777778,
        "bt_art_rate_delta": 0.125,
        "bt_art_rate_delta_ci_95": [
          0.013888888888888881,
          0.25
        ],
        "bt_art_rate_treated": 0.27777777777777773,
        "n_prompt_pairs": 24,
        "rv_cohens_dz": -0.06985455437693215,
        "rv_delta_ci_95": [
          -0.05566309175793953,
          0.03948728655535559
        ],
        "rv_delta_mean": -0.00860204978551969,
        "rv_p_value": 0.7352957850489128
      },
      "early_mlp_0p0625_bridge_2": {
        "alpha": 2.0625,
        "bt_art_cohens_h": 0.037919865181064494,
        "bt_art_exact_sign_p": 0.6875,
        "bt_art_prompt_losses": 2,
        "bt_art_prompt_wins": 4,
        "bt_art_rate_control": 0.1527777777777778,
        "bt_art_rate_delta": 0.01388888888888889,
        "bt_art_rate_delta_ci_95": [
          -0.11111111111111112,
          0.125
        ],
        "bt_art_rate_treated": 0.16666666666666666,
        "n_prompt_pairs": 24,
        "rv_cohens_dz": -0.10720456221518468,
        "rv_delta_ci_95": [
          -0.05800389778930902,
          0.03316441548407276
        ],
        "rv_delta_mean": -0.012645768575731792,
        "rv_p_value": 0.6044755794314369
      },
      "early_mlp_0p0625_bridge_3": {
        "alpha": 3.0625,
        "bt_art_cohens_h": 0.2758374268425864,
        "bt_art_exact_sign_p": 0.09228515625,
        "bt_art_prompt_losses": 3,
        "bt_art_prompt_wins": 10,
        "bt_art_rate_control": 0.1527777777777778,
        "bt_art_rate_delta": 0.11111111111111109,
        "bt_art_rate_delta_ci_95": [
          -0.0138888888888889,
          0.23611111111111108
        ],
        "bt_art_rate_treated": 0.2638888888888889,
        "n_prompt_pairs": 24,
        "rv_cohens_dz": -0.04717306730908512,
        "rv_delta_ci_95": [
          -0.05601842126470849,
          0.043716070979350305
        ],
        "rv_delta_mean": -0.006099486663845455,
        "rv_p_value": 0.8192806678926714
      },
      "early_mlp_0p125_bridge_2": {
        "alpha": 2.125,
        "bt_art_cohens_h": 0.21166536888748866,
        "bt_art_exact_sign_p": 0.266845703125,
        "bt_art_prompt_losses": 4,
        "bt_art_prompt_wins": 9,
        "bt_art_rate_control": 0.1527777777777778,
        "bt_art_rate_delta": 0.08333333333333333,
        "bt_art_rate_delta_ci_95": [
          -0.04166666666666668,
          0.20833333333333334
        ],
        "bt_art_rate_treated": 0.23611111111111108,
        "n_prompt_pairs": 24,
        "rv_cohens_dz": -0.09432240567081011,
        "rv_delta_ci_95": [
          -0.04953242238123568,
          0.029672662686707468
        ],
        "rv_delta_mean": -0.009303704177089719,
        "rv_p_value": 0.6483626591651952
      },
      "early_mlp_0p125_bridge_3": {
        "alpha": 3.125,
        "bt_art_cohens_h": 0.3070935297267082,
        "bt_art_exact_sign_p": 0.2265625,
        "bt_art_prompt_losses": 3,
        "bt_art_prompt_wins": 8,
        "bt_art_rate_control": 0.1527777777777778,
        "bt_art_rate_delta": 0.125,
        "bt_art_rate_delta_ci_95": [
          0.013888888888888876,
          0.25
        ],
        "bt_art_rate_treated": 0.27777777777777773,
        "n_prompt_pairs": 24,
        "rv_cohens_dz": -0.23223077446624749,
        "rv_delta_ci_95": [
          -0.05995981987400341,
          0.014626430590057788
        ],
        "rv_delta_mean": -0.021818262868198893,
        "rv_p_value": 0.26696177671980165
      },
      "early_mlp_0p1875_bridge_2": {
        "alpha": 2.1875,
        "bt_art_cohens_h": 0.2758374268425864,
        "bt_art_exact_sign_p": 0.1796875,
        "bt_art_prompt_losses": 2,
        "bt_art_prompt_wins": 7,
        "bt_art_rate_control": 0.1527777777777778,
        "bt_art_rate_delta": 0.1111111111111111,
        "bt_art_rate_delta_ci_95": [
          0.013888888888888888,
          0.2222222222222222
        ],
        "bt_art_rate_treated": 0.2638888888888889,
        "n_prompt_pairs": 24,
        "rv_cohens_dz": -0.0838185116168817,
        "rv_delta_ci_95": [
          -0.05817349866315555,
          0.03851715681106371
        ],
        "rv_delta_mean": -0.010258515457029008,
        "rv_p_value": 0.6851477566269047
      },
      "early_mlp_0p1875_bridge_3": {
        "alpha": 3.1875,
        "bt_art_cohens_h": 0.27583742684258616,
        "bt_art_exact_sign_p": 0.0390625,
        "bt_art_prompt_losses": 1,
        "bt_art_prompt_wins": 8,
        "bt_art_rate_control": 0.1527777777777778,
        "bt_art_rate_delta": 0.1111111111111111,
        "bt_art_rate_delta_ci_95": [
          0.02777777777777778,
          0.19444444444444445
        ],
        "bt_art_rate_treated": 0.26388888888888884,
        "n_prompt_pairs": 24,
        "rv_cohens_dz": -0.18173733101620146,
        "rv_delta_ci_95": [
          -0.053887125113092474,
          0.019473619811573552
        ],
        "rv_delta_mean": -0.016753131424258903,
        "rv_p_value": 0.38250886860155164
      }
    },
    "recursive": {
      "bridge_only_2": {
        "alpha": 2.0,
        "bt_art_cohens_h": -0.20612888062866763,
        "bt_art_exact_sign_p": 0.625,
        "bt_art_prompt_losses": 3,
        "bt_art_prompt_wins": 1,
        "bt_art_rate_control": 0.25,
        "bt_art_rate_delta": -0.08333333333333333,
        "bt_art_rate_delta_ci_95": [
          -0.22222222222222224,
          0.05555555555555555
        ],
        "bt_art_rate_treated": 0.16666666666666666,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": 0.10286869794034421,
        "rv_delta_ci_95": [
          -0.06152799434803078,
          0.09158745769631572
        ],
        "rv_delta_mean": 0.014287971907574882,
        "rv_p_value": 0.7283210893479407
      },
      "bridge_only_3": {
        "alpha": 3.0,
        "bt_art_cohens_h": 0.4122577612573348,
        "bt_art_exact_sign_p": 0.109375,
        "bt_art_prompt_losses": 2,
        "bt_art_prompt_wins": 8,
        "bt_art_rate_control": 0.25,
        "bt_art_rate_delta": 0.19444444444444442,
        "bt_art_rate_delta_ci_95": [
          0.027777777777777762,
          0.3611111111111111
        ],
        "bt_art_rate_treated": 0.44444444444444436,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": -0.11529648163152986,
        "rv_delta_ci_95": [
          -0.08429765971704818,
          0.04601212296096511
        ],
        "rv_delta_mean": -0.013995962731565267,
        "rv_p_value": 0.6972446867489017
      },
      "early_mlp_0p03125_bridge_2": {
        "alpha": 2.03125,
        "bt_art_cohens_h": -0.06543219461797511,
        "bt_art_exact_sign_p": 1.0,
        "bt_art_prompt_losses": 2,
        "bt_art_prompt_wins": 3,
        "bt_art_rate_control": 0.25,
        "bt_art_rate_delta": -0.027777777777777787,
        "bt_art_rate_delta_ci_95": [
          -0.25,
          0.13888888888888887
        ],
        "bt_art_rate_treated": 0.2222222222222222,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": -0.05271355622600459,
        "rv_delta_ci_95": [
          -0.09006801576798944,
          0.07051409428236512
        ],
        "rv_delta_mean": -0.007795847129758772,
        "rv_p_value": 0.8584297796702294
      },
      "early_mlp_0p03125_bridge_3": {
        "alpha": 3.03125,
        "bt_art_cohens_h": 0.579182948879216,
        "bt_art_exact_sign_p": 0.03857421875,
        "bt_art_prompt_losses": 2,
        "bt_art_prompt_wins": 10,
        "bt_art_rate_control": 0.25,
        "bt_art_rate_delta": 0.27777777777777785,
        "bt_art_rate_delta_ci_95": [
          0.11111111111111109,
          0.4444444444444444
        ],
        "bt_art_rate_treated": 0.5277777777777777,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": -0.1723590825064584,
        "rv_delta_ci_95": [
          -0.09995168041261082,
          0.04802706305704572
        ],
        "rv_delta_mean": -0.023517759956343264,
        "rv_p_value": 0.5625490275658986
      },
      "early_mlp_0p0625_bridge_2": {
        "alpha": 2.0625,
        "bt_art_cohens_h": 0.0630447839169761,
        "bt_art_exact_sign_p": 0.6875,
        "bt_art_prompt_losses": 2,
        "bt_art_prompt_wins": 4,
        "bt_art_rate_control": 0.25,
        "bt_art_rate_delta": 0.027777777777777776,
        "bt_art_rate_delta_ci_95": [
          -0.2222222222222222,
          0.22222222222222224
        ],
        "bt_art_rate_treated": 0.27777777777777773,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": -0.12491087796626153,
        "rv_delta_ci_95": [
          -0.09109371830606307,
          0.0525016117147578
        ],
        "rv_delta_mean": -0.016578534476727354,
        "rv_p_value": 0.6735913044072267
      },
      "early_mlp_0p0625_bridge_3": {
        "alpha": 3.0625,
        "bt_art_cohens_h": 0.4122577612573348,
        "bt_art_exact_sign_p": 0.0654296875,
        "bt_art_prompt_losses": 2,
        "bt_art_prompt_wins": 9,
        "bt_art_rate_control": 0.25,
        "bt_art_rate_delta": 0.19444444444444442,
        "bt_art_rate_delta_ci_95": [
          -0.027777777777777794,
          0.36111111111111116
        ],
        "bt_art_rate_treated": 0.44444444444444436,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": -0.12678632819435415,
        "rv_delta_ci_95": [
          -0.09858707842107779,
          0.05509878454910252
        ],
        "rv_delta_mean": -0.01783476490600797,
        "rv_p_value": 0.6690189314224495
      },
      "early_mlp_0p125_bridge_2": {
        "alpha": 2.125,
        "bt_art_cohens_h": 0.3561506963786092,
        "bt_art_exact_sign_p": 0.1796875,
        "bt_art_prompt_losses": 2,
        "bt_art_prompt_wins": 7,
        "bt_art_rate_control": 0.25,
        "bt_art_rate_delta": 0.16666666666666666,
        "bt_art_rate_delta_ci_95": [
          -0.05555555555555556,
          0.36111111111111116
        ],
        "bt_art_rate_treated": 0.4166666666666666,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": -0.14399782529881125,
        "rv_delta_ci_95": [
          -0.08646514418607329,
          0.045136687567997054
        ],
        "rv_delta_mean": -0.017109726307163422,
        "rv_p_value": 0.6277286363976836
      },
      "early_mlp_0p125_bridge_3": {
        "alpha": 3.125,
        "bt_art_cohens_h": 0.4122577612573348,
        "bt_art_exact_sign_p": 0.2890625,
        "bt_art_prompt_losses": 2,
        "bt_art_prompt_wins": 6,
        "bt_art_rate_control": 0.25,
        "bt_art_rate_delta": 0.19444444444444442,
        "bt_art_rate_delta_ci_95": [
          -9.25185853854297e-18,
          0.3888888888888889
        ],
        "bt_art_rate_treated": 0.4444444444444444,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": -0.573377749280922,
        "rv_delta_ci_95": [
          -0.11573560705696612,
          -0.004607690447837069
        ],
        "rv_delta_mean": -0.05811585729453372,
        "rv_p_value": 0.07249180255208958
      },
      "early_mlp_0p1875_bridge_2": {
        "alpha": 2.1875,
        "bt_art_cohens_h": 0.4122577612573348,
        "bt_art_exact_sign_p": 0.125,
        "bt_art_prompt_losses": 1,
        "bt_art_prompt_wins": 6,
        "bt_art_rate_control": 0.25,
        "bt_art_rate_delta": 0.19444444444444445,
        "bt_art_rate_delta_ci_95": [
          0.027777777777777787,
          0.3611111111111111
        ],
        "bt_art_rate_treated": 0.44444444444444436,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": -0.2579933907518902,
        "rv_delta_ci_95": [
          -0.11190664378545273,
          0.03367104820349632
        ],
        "rv_delta_mean": -0.03434795840621696,
        "rv_p_value": 0.3906087578035241
      },
      "early_mlp_0p1875_bridge_3": {
        "alpha": 3.1875,
        "bt_art_cohens_h": 0.4122577612573348,
        "bt_art_exact_sign_p": 0.03125,
        "bt_art_prompt_losses": 0,
        "bt_art_prompt_wins": 6,
        "bt_art_rate_control": 0.25,
        "bt_art_rate_delta": 0.19444444444444445,
        "bt_art_rate_delta_ci_95": [
          0.08333333333333333,
          0.30555555555555564
        ],
        "bt_art_rate_treated": 0.44444444444444436,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": -0.3427085614069515,
        "rv_delta_ci_95": [
          -0.10414992660264982,
          0.02080807688280151
        ],
        "rv_delta_mean": -0.03976544826779776,
        "rv_p_value": 0.2601626030783893
      }
    }
  },
  "experiment": "causal_state_benchmark_v4_multisite",
  "generation_seeds": [
    101,
    202,
    303
  ],
  "heldout_prompt_counts": {
    "by_group": {
      "L3_deeper": 4,
      "L4_full": 4,
      "L5_refined": 4,
      "baseline_creative": 4,
      "baseline_factual": 4,
      "baseline_math": 4
    },
    "by_mode": {
      "baseline": 12,
      "recursive": 12
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
        "bridge": 2.0,
        "early_mlp": 0.0
      },
      "name": "bridge_only_2"
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
        "bridge": 2.0,
        "early_mlp": 0.03125
      },
      "name": "early_mlp_0p03125_bridge_2"
    },
    {
      "alphas": {
        "bridge": 2.0,
        "early_mlp": 0.0625
      },
      "name": "early_mlp_0p0625_bridge_2"
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
        "bridge": 2.0,
        "early_mlp": 0.1875
      },
      "name": "early_mlp_0p1875_bridge_2"
    },
    {
      "alphas": {
        "bridge": 3.0,
        "early_mlp": 0.03125
      },
      "name": "early_mlp_0p03125_bridge_3"
    },
    {
      "alphas": {
        "bridge": 3.0,
        "early_mlp": 0.0625
      },
      "name": "early_mlp_0p0625_bridge_3"
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
        "early_mlp": 0.1875
      },
      "name": "early_mlp_0p1875_bridge_3"
    }
  ],
  "n_generation_seeds": 3,
  "n_pairs": 24,
  "n_total": 24,
  "primary_prompt_mode": "recursive",
  "prompt_bank_version": "2ac959a313614329",
  "schema_version": "metrics_summary_v1",
  "source_layers": {
    "bridge": {
      "centroid_cosine": 0.8922646045684814,
      "component": "residual",
      "direction_norm": 5.84972620010376,
      "layer": 25,
      "token_window": null,
      "window": 32
    },
    "early_mlp": {
      "centroid_cosine": 0.7239384651184082,
      "component": "mlp",
      "direction_norm": 0.10180021077394485,
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
  "timestamp": "20260312_141450",
  "top_p": 0.95,
  "verdict": "multisite_additive"
}
```
