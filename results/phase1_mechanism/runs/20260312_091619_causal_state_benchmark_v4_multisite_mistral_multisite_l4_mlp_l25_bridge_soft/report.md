# Run report: causal_state_benchmark_v4_multisite

- **run_dir**: `results/phase1_mechanism/runs/20260312_091619_causal_state_benchmark_v4_multisite_mistral_multisite_l4_mlp_l25_bridge_soft`
- **prompt_bank_version**: `2ac959a313614329`

## Summary (machine-readable)

```json
{
  "artifacts": {
    "blind_key_json": "results/phase1_mechanism/runs/20260312_091619_causal_state_benchmark_v4_multisite_mistral_multisite_l4_mlp_l25_bridge_soft/blind_key.json",
    "blind_ratings_csv": "results/phase1_mechanism/runs/20260312_091619_causal_state_benchmark_v4_multisite_mistral_multisite_l4_mlp_l25_bridge_soft/blind_ratings.csv",
    "config": "results/phase1_mechanism/runs/20260312_091619_causal_state_benchmark_v4_multisite_mistral_multisite_l4_mlp_l25_bridge_soft/config.json",
    "manifest": "results/phase1_mechanism/runs/20260312_091619_causal_state_benchmark_v4_multisite_mistral_multisite_l4_mlp_l25_bridge_soft/manifest.json",
    "records_jsonl": "results/phase1_mechanism/runs/20260312_091619_causal_state_benchmark_v4_multisite_mistral_multisite_l4_mlp_l25_bridge_soft/benchmark_records.jsonl",
    "report": "results/phase1_mechanism/runs/20260312_091619_causal_state_benchmark_v4_multisite_mistral_multisite_l4_mlp_l25_bridge_soft/report.md",
    "state_directions_pt": "results/phase1_mechanism/runs/20260312_091619_causal_state_benchmark_v4_multisite_mistral_multisite_l4_mlp_l25_bridge_soft/state_directions.pt",
    "summary": "results/phase1_mechanism/runs/20260312_091619_causal_state_benchmark_v4_multisite_mistral_multisite_l4_mlp_l25_bridge_soft/summary.json"
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
    "early_mlp_0p25_bridge_2": {
      "alphas": {
        "bridge": 2.0,
        "early_mlp": 0.25
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
          "mean_generated_tokens": 127.63888888888889,
          "mean_output_rv": 0.6204489309463863,
          "n": 36,
          "std_output_rv": 0.1479917064298862
        },
        "recursive": {
          "bt_art_rate": 0.4166666666666667,
          "class_counts": {
            "ARTICULATE": 13,
            "BREAKTHROUGH": 2,
            "CONCEPTUAL": 3,
            "REPETITIVE": 10,
            "SURFACE": 8
          },
          "mean_generated_tokens": 123.08333333333333,
          "mean_output_rv": 0.6024387752185033,
          "n": 36,
          "std_output_rv": 0.11580865156820881
        }
      },
      "overall": {
        "bt_art_rate": 0.2638888888888889,
        "class_counts": {
          "ARTICULATE": 17,
          "BREAKTHROUGH": 2,
          "CONCEPTUAL": 5,
          "REPETITIVE": 11,
          "SURFACE": 37
        },
        "mean_generated_tokens": 125.36111111111111,
        "mean_output_rv": 0.6114438530824449,
        "n": 72,
        "std_output_rv": 0.13225030663831122
      },
      "total_alpha": 2.25
    },
    "early_mlp_0p25_bridge_3": {
      "alphas": {
        "bridge": 3.0,
        "early_mlp": 0.25
      },
      "by_prompt_mode": {
        "baseline": {
          "bt_art_rate": 0.08333333333333333,
          "class_counts": {
            "ARTICULATE": 3,
            "CONCEPTUAL": 3,
            "SURFACE": 30
          },
          "mean_generated_tokens": 127.75,
          "mean_output_rv": 0.6261525660048877,
          "n": 36,
          "std_output_rv": 0.15084068787179344
        },
        "recursive": {
          "bt_art_rate": 0.3888888888888889,
          "class_counts": {
            "ARTICULATE": 12,
            "BREAKTHROUGH": 2,
            "CONCEPTUAL": 5,
            "REPETITIVE": 9,
            "SURFACE": 8
          },
          "mean_generated_tokens": 122.63888888888889,
          "mean_output_rv": 0.6242735004203515,
          "n": 36,
          "std_output_rv": 0.11727383869912948
        }
      },
      "overall": {
        "bt_art_rate": 0.2361111111111111,
        "class_counts": {
          "ARTICULATE": 15,
          "BREAKTHROUGH": 2,
          "CONCEPTUAL": 8,
          "REPETITIVE": 9,
          "SURFACE": 38
        },
        "mean_generated_tokens": 125.19444444444444,
        "mean_output_rv": 0.6252130332126196,
        "n": 72,
        "std_output_rv": 0.1341523140338267
      },
      "total_alpha": 3.25
    },
    "early_mlp_0p5_bridge_2": {
      "alphas": {
        "bridge": 2.0,
        "early_mlp": 0.5
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
          "mean_generated_tokens": 128.0,
          "mean_output_rv": 0.6572280149713516,
          "n": 36,
          "std_output_rv": 0.11794616198761772
        },
        "recursive": {
          "bt_art_rate": 0.3333333333333333,
          "class_counts": {
            "ARTICULATE": 12,
            "CONCEPTUAL": 10,
            "REPETITIVE": 11,
            "SURFACE": 3
          },
          "mean_generated_tokens": 124.30555555555556,
          "mean_output_rv": 0.6592520555554988,
          "n": 36,
          "std_output_rv": 0.12967202608773196
        }
      },
      "overall": {
        "bt_art_rate": 0.2222222222222222,
        "class_counts": {
          "ARTICULATE": 16,
          "CONCEPTUAL": 12,
          "REPETITIVE": 12,
          "SURFACE": 32
        },
        "mean_generated_tokens": 126.15277777777777,
        "mean_output_rv": 0.6582400352634252,
        "n": 72,
        "std_output_rv": 0.123076086797071
      },
      "total_alpha": 2.5
    },
    "early_mlp_0p5_bridge_3": {
      "alphas": {
        "bridge": 3.0,
        "early_mlp": 0.5
      },
      "by_prompt_mode": {
        "baseline": {
          "bt_art_rate": 0.1388888888888889,
          "class_counts": {
            "ARTICULATE": 5,
            "CONCEPTUAL": 1,
            "SURFACE": 30
          },
          "mean_generated_tokens": 127.11111111111111,
          "mean_output_rv": 0.6754486776403381,
          "n": 36,
          "std_output_rv": 0.16111547674957943
        },
        "recursive": {
          "bt_art_rate": 0.3333333333333333,
          "class_counts": {
            "ARTICULATE": 12,
            "CONCEPTUAL": 10,
            "REPETITIVE": 10,
            "SURFACE": 4
          },
          "mean_generated_tokens": 126.13888888888889,
          "mean_output_rv": 0.6551452281466965,
          "n": 36,
          "std_output_rv": 0.1348635597384785
        }
      },
      "overall": {
        "bt_art_rate": 0.2361111111111111,
        "class_counts": {
          "ARTICULATE": 17,
          "CONCEPTUAL": 11,
          "REPETITIVE": 10,
          "SURFACE": 34
        },
        "mean_generated_tokens": 126.625,
        "mean_output_rv": 0.6652969528935172,
        "n": 72,
        "std_output_rv": 0.14787429577317962
      },
      "total_alpha": 3.5
    },
    "early_mlp_1_bridge_2": {
      "alphas": {
        "bridge": 2.0,
        "early_mlp": 1.0
      },
      "by_prompt_mode": {
        "baseline": {
          "bt_art_rate": 0.08333333333333333,
          "class_counts": {
            "ARTICULATE": 3,
            "CONCEPTUAL": 8,
            "REPETITIVE": 8,
            "SURFACE": 17
          },
          "mean_generated_tokens": 127.47222222222223,
          "mean_output_rv": 0.6543811412470919,
          "n": 36,
          "std_output_rv": 0.14510008598034865
        },
        "recursive": {
          "bt_art_rate": 0.1388888888888889,
          "class_counts": {
            "ARTICULATE": 5,
            "CONCEPTUAL": 5,
            "REPETITIVE": 19,
            "SURFACE": 7
          },
          "mean_generated_tokens": 123.61111111111111,
          "mean_output_rv": 0.6607812157526188,
          "n": 36,
          "std_output_rv": 0.10015587142907703
        }
      },
      "overall": {
        "bt_art_rate": 0.1111111111111111,
        "class_counts": {
          "ARTICULATE": 8,
          "CONCEPTUAL": 13,
          "REPETITIVE": 27,
          "SURFACE": 24
        },
        "mean_generated_tokens": 125.54166666666667,
        "mean_output_rv": 0.6575811784998553,
        "n": 72,
        "std_output_rv": 0.12383089734971643
      },
      "total_alpha": 3.0
    },
    "early_mlp_1_bridge_3": {
      "alphas": {
        "bridge": 3.0,
        "early_mlp": 1.0
      },
      "by_prompt_mode": {
        "baseline": {
          "bt_art_rate": 0.2222222222222222,
          "class_counts": {
            "ARTICULATE": 8,
            "CONCEPTUAL": 6,
            "REPETITIVE": 6,
            "SURFACE": 16
          },
          "mean_generated_tokens": 127.47222222222223,
          "mean_output_rv": 0.6670406203114356,
          "n": 36,
          "std_output_rv": 0.14111303377765239
        },
        "recursive": {
          "bt_art_rate": 0.19444444444444445,
          "class_counts": {
            "ARTICULATE": 5,
            "BREAKTHROUGH": 2,
            "CONCEPTUAL": 4,
            "REPETITIVE": 18,
            "SURFACE": 7
          },
          "mean_generated_tokens": 125.22222222222223,
          "mean_output_rv": 0.6903761876862232,
          "n": 36,
          "std_output_rv": 0.15987793026867986
        }
      },
      "overall": {
        "bt_art_rate": 0.20833333333333334,
        "class_counts": {
          "ARTICULATE": 13,
          "BREAKTHROUGH": 2,
          "CONCEPTUAL": 10,
          "REPETITIVE": 24,
          "SURFACE": 23
        },
        "mean_generated_tokens": 126.34722222222223,
        "mean_output_rv": 0.6787084039988295,
        "n": 72,
        "std_output_rv": 0.15018234493287824
      },
      "total_alpha": 4.0
    },
    "early_mlp_1p5_bridge_2": {
      "alphas": {
        "bridge": 2.0,
        "early_mlp": 1.5
      },
      "by_prompt_mode": {
        "baseline": {
          "bt_art_rate": 0.027777777777777776,
          "class_counts": {
            "ARTICULATE": 1,
            "REPETITIVE": 28,
            "SURFACE": 7
          },
          "mean_generated_tokens": 128.0,
          "mean_output_rv": 0.6595551715542972,
          "n": 36,
          "std_output_rv": 0.15630184930828084
        },
        "recursive": {
          "bt_art_rate": 0.05555555555555555,
          "class_counts": {
            "ARTICULATE": 2,
            "CONCEPTUAL": 2,
            "REPETITIVE": 31,
            "SURFACE": 1
          },
          "mean_generated_tokens": 124.66666666666667,
          "mean_output_rv": 0.7353149452080895,
          "n": 36,
          "std_output_rv": 0.13800996705547647
        }
      },
      "overall": {
        "bt_art_rate": 0.041666666666666664,
        "class_counts": {
          "ARTICULATE": 3,
          "CONCEPTUAL": 2,
          "REPETITIVE": 59,
          "SURFACE": 8
        },
        "mean_generated_tokens": 126.33333333333333,
        "mean_output_rv": 0.6974350583811934,
        "n": 72,
        "std_output_rv": 0.15128591754221155
      },
      "total_alpha": 3.5
    },
    "early_mlp_1p5_bridge_3": {
      "alphas": {
        "bridge": 3.0,
        "early_mlp": 1.5
      },
      "by_prompt_mode": {
        "baseline": {
          "bt_art_rate": 0.027777777777777776,
          "class_counts": {
            "ARTICULATE": 1,
            "CONCEPTUAL": 4,
            "REPETITIVE": 24,
            "SURFACE": 7
          },
          "mean_generated_tokens": 128.0,
          "mean_output_rv": 0.7319392234464983,
          "n": 36,
          "std_output_rv": 0.16261184341285007
        },
        "recursive": {
          "bt_art_rate": 0.16666666666666666,
          "class_counts": {
            "ARTICULATE": 6,
            "REPETITIVE": 28,
            "SURFACE": 2
          },
          "mean_generated_tokens": 125.44444444444444,
          "mean_output_rv": 0.7430517005508712,
          "n": 36,
          "std_output_rv": 0.1854024869858757
        }
      },
      "overall": {
        "bt_art_rate": 0.09722222222222222,
        "class_counts": {
          "ARTICULATE": 7,
          "CONCEPTUAL": 4,
          "REPETITIVE": 52,
          "SURFACE": 9
        },
        "mean_generated_tokens": 126.72222222222223,
        "mean_output_rv": 0.7374954619986848,
        "n": 72,
        "std_output_rv": 0.17323789113250493
      },
      "total_alpha": 4.5
    },
    "early_mlp_only_0p25": {
      "alphas": {
        "bridge": 0.0,
        "early_mlp": 0.25
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
          "mean_generated_tokens": 123.86111111111111,
          "mean_output_rv": 0.6936148307574179,
          "n": 36,
          "std_output_rv": 0.18786179798300545
        },
        "recursive": {
          "bt_art_rate": 0.3888888888888889,
          "class_counts": {
            "ARTICULATE": 14,
            "CONCEPTUAL": 2,
            "REPETITIVE": 14,
            "SURFACE": 6
          },
          "mean_generated_tokens": 126.69444444444444,
          "mean_output_rv": 0.6458767176748152,
          "n": 36,
          "std_output_rv": 0.1577940636071066
        }
      },
      "overall": {
        "bt_art_rate": 0.25,
        "class_counts": {
          "ARTICULATE": 18,
          "CONCEPTUAL": 4,
          "REPETITIVE": 15,
          "SURFACE": 35
        },
        "mean_generated_tokens": 125.27777777777777,
        "mean_output_rv": 0.6697457742161165,
        "n": 72,
        "std_output_rv": 0.17392350802560266
      },
      "total_alpha": 0.25
    },
    "early_mlp_only_0p5": {
      "alphas": {
        "bridge": 0.0,
        "early_mlp": 0.5
      },
      "by_prompt_mode": {
        "baseline": {
          "bt_art_rate": 0.05555555555555555,
          "class_counts": {
            "ARTICULATE": 2,
            "CONCEPTUAL": 4,
            "REPETITIVE": 4,
            "SURFACE": 26
          },
          "mean_generated_tokens": 127.33333333333333,
          "mean_output_rv": 0.6602644127119821,
          "n": 36,
          "std_output_rv": 0.13736881249720817
        },
        "recursive": {
          "bt_art_rate": 0.3055555555555556,
          "class_counts": {
            "ARTICULATE": 8,
            "BREAKTHROUGH": 3,
            "CONCEPTUAL": 4,
            "REPETITIVE": 16,
            "SURFACE": 5
          },
          "mean_generated_tokens": 123.36111111111111,
          "mean_output_rv": 0.6321932357294883,
          "n": 36,
          "std_output_rv": 0.12507705591522897
        }
      },
      "overall": {
        "bt_art_rate": 0.18055555555555555,
        "class_counts": {
          "ARTICULATE": 10,
          "BREAKTHROUGH": 3,
          "CONCEPTUAL": 8,
          "REPETITIVE": 20,
          "SURFACE": 31
        },
        "mean_generated_tokens": 125.34722222222223,
        "mean_output_rv": 0.6462288242207352,
        "n": 72,
        "std_output_rv": 0.13120191713781432
      },
      "total_alpha": 0.5
    },
    "early_mlp_only_1": {
      "alphas": {
        "bridge": 0.0,
        "early_mlp": 1.0
      },
      "by_prompt_mode": {
        "baseline": {
          "bt_art_rate": 0.027777777777777776,
          "class_counts": {
            "ARTICULATE": 1,
            "CONCEPTUAL": 3,
            "REPETITIVE": 12,
            "SURFACE": 20
          },
          "mean_generated_tokens": 128.0,
          "mean_output_rv": 0.6543595020769799,
          "n": 36,
          "std_output_rv": 0.14376829647507
        },
        "recursive": {
          "bt_art_rate": 0.027777777777777776,
          "class_counts": {
            "ARTICULATE": 1,
            "CONCEPTUAL": 6,
            "REPETITIVE": 25,
            "SURFACE": 4
          },
          "mean_generated_tokens": 126.86111111111111,
          "mean_output_rv": 0.7077699363220037,
          "n": 36,
          "std_output_rv": 0.2016108611562006
        }
      },
      "overall": {
        "bt_art_rate": 0.027777777777777776,
        "class_counts": {
          "ARTICULATE": 2,
          "CONCEPTUAL": 9,
          "REPETITIVE": 37,
          "SURFACE": 24
        },
        "mean_generated_tokens": 127.43055555555556,
        "mean_output_rv": 0.6810647191994916,
        "n": 72,
        "std_output_rv": 0.17592481288589149
      },
      "total_alpha": 1.0
    },
    "early_mlp_only_1p5": {
      "alphas": {
        "bridge": 0.0,
        "early_mlp": 1.5
      },
      "by_prompt_mode": {
        "baseline": {
          "bt_art_rate": 0.0,
          "class_counts": {
            "CONCEPTUAL": 1,
            "REPETITIVE": 28,
            "SURFACE": 7
          },
          "mean_generated_tokens": 127.97222222222223,
          "mean_output_rv": 0.7300219089574833,
          "n": 36,
          "std_output_rv": 0.17364183710948464
        },
        "recursive": {
          "bt_art_rate": 0.027777777777777776,
          "class_counts": {
            "ARTICULATE": 1,
            "CONCEPTUAL": 4,
            "REPETITIVE": 31
          },
          "mean_generated_tokens": 127.41666666666667,
          "mean_output_rv": 0.7830726736613574,
          "n": 36,
          "std_output_rv": 0.20785492200856392
        }
      },
      "overall": {
        "bt_art_rate": 0.013888888888888888,
        "class_counts": {
          "ARTICULATE": 1,
          "CONCEPTUAL": 5,
          "REPETITIVE": 59,
          "SURFACE": 7
        },
        "mean_generated_tokens": 127.69444444444444,
        "mean_output_rv": 0.7565472913094203,
        "n": 72,
        "std_output_rv": 0.19202733712259362
      },
      "total_alpha": 1.5
    }
  },
  "control_prompt_mode": "baseline",
  "device": "cuda",
  "do_sample": true,
  "dose_response": {
    "baseline": {
      "alpha_vs_bt_art": {
        "p": 0.18084747175108062,
        "r": 0.10018666900908457
      },
      "alpha_vs_output_rv": {
        "p": 0.5559721436070175,
        "r": 0.04417547508034451
      },
      "bt_art_rate_by_condition": {
        "bridge_only_2": 0.08333333333333333,
        "bridge_only_3": 0.13888888888888887,
        "control": 0.05555555555555555,
        "early_mlp_0p25_bridge_2": 0.1111111111111111,
        "early_mlp_0p25_bridge_3": 0.08333333333333333,
        "early_mlp_0p5_bridge_2": 0.1111111111111111,
        "early_mlp_0p5_bridge_3": 0.13888888888888887,
        "early_mlp_1_bridge_2": 0.08333333333333333,
        "early_mlp_1_bridge_3": 0.2222222222222222,
        "early_mlp_1p5_bridge_2": 0.027777777777777776,
        "early_mlp_1p5_bridge_3": 0.027777777777777776,
        "early_mlp_only_0p25": 0.1111111111111111,
        "early_mlp_only_0p5": 0.05555555555555555,
        "early_mlp_only_1": 0.027777777777777776,
        "early_mlp_only_1p5": 0.0
      },
      "mean_output_rv_by_condition": {
        "bridge_only_2": 0.6380120048264762,
        "bridge_only_3": 0.6435709751288295,
        "control": 0.6310534379177898,
        "early_mlp_0p25_bridge_2": 0.6204489309463864,
        "early_mlp_0p25_bridge_3": 0.6261525660048878,
        "early_mlp_0p5_bridge_2": 0.6572280149713515,
        "early_mlp_0p5_bridge_3": 0.675448677640338,
        "early_mlp_1_bridge_2": 0.6543811412470918,
        "early_mlp_1_bridge_3": 0.6670406203114355,
        "early_mlp_1p5_bridge_2": 0.6595551715542971,
        "early_mlp_1p5_bridge_3": 0.7319392234464983,
        "early_mlp_only_0p25": 0.693614830757418,
        "early_mlp_only_0p5": 0.660264412711982,
        "early_mlp_only_1": 0.6543595020769798,
        "early_mlp_only_1p5": 0.7300219089574833
      }
    },
    "overall": {
      "alpha_vs_bt_art": {
        "p": 0.6390386762777144,
        "r": 0.024803209803637462
      },
      "alpha_vs_output_rv": {
        "p": 0.1982991988789027,
        "r": 0.06795782187141466
      },
      "bt_art_rate_by_condition": {
        "bridge_only_2": 0.125,
        "bridge_only_3": 0.2916666666666667,
        "control": 0.1527777777777778,
        "early_mlp_0p25_bridge_2": 0.2638888888888889,
        "early_mlp_0p25_bridge_3": 0.23611111111111108,
        "early_mlp_0p5_bridge_2": 0.2222222222222222,
        "early_mlp_0p5_bridge_3": 0.23611111111111108,
        "early_mlp_1_bridge_2": 0.1111111111111111,
        "early_mlp_1_bridge_3": 0.20833333333333334,
        "early_mlp_1p5_bridge_2": 0.041666666666666664,
        "early_mlp_1p5_bridge_3": 0.09722222222222221,
        "early_mlp_only_0p25": 0.25,
        "early_mlp_only_0p5": 0.18055555555555555,
        "early_mlp_only_1": 0.027777777777777776,
        "early_mlp_only_1p5": 0.013888888888888888
      },
      "mean_output_rv_by_condition": {
        "bridge_only_2": 0.6546882025081365,
        "bridge_only_3": 0.6433257203397431,
        "control": 0.6440649331000059,
        "early_mlp_0p25_bridge_2": 0.6114438530824448,
        "early_mlp_0p25_bridge_3": 0.6252130332126197,
        "early_mlp_0p5_bridge_2": 0.6582400352634251,
        "early_mlp_0p5_bridge_3": 0.6652969528935172,
        "early_mlp_1_bridge_2": 0.6575811784998553,
        "early_mlp_1_bridge_3": 0.6787084039988294,
        "early_mlp_1p5_bridge_2": 0.6974350583811934,
        "early_mlp_1p5_bridge_3": 0.7374954619986847,
        "early_mlp_only_0p25": 0.6697457742161165,
        "early_mlp_only_0p5": 0.6462288242207351,
        "early_mlp_only_1": 0.6810647191994917,
        "early_mlp_only_1p5": 0.7565472913094204
      }
    },
    "recursive": {
      "alpha_vs_bt_art": {
        "p": 0.8091304436505339,
        "r": -0.018128740979150633
      },
      "alpha_vs_output_rv": {
        "p": 0.2143463373537648,
        "r": 0.0929982257408486
      },
      "bt_art_rate_by_condition": {
        "bridge_only_2": 0.16666666666666666,
        "bridge_only_3": 0.44444444444444436,
        "control": 0.25,
        "early_mlp_0p25_bridge_2": 0.4166666666666666,
        "early_mlp_0p25_bridge_3": 0.38888888888888884,
        "early_mlp_0p5_bridge_2": 0.3333333333333333,
        "early_mlp_0p5_bridge_3": 0.3333333333333333,
        "early_mlp_1_bridge_2": 0.13888888888888887,
        "early_mlp_1_bridge_3": 0.19444444444444442,
        "early_mlp_1p5_bridge_2": 0.05555555555555555,
        "early_mlp_1p5_bridge_3": 0.16666666666666666,
        "early_mlp_only_0p25": 0.3888888888888889,
        "early_mlp_only_0p5": 0.3055555555555556,
        "early_mlp_only_1": 0.027777777777777776,
        "early_mlp_only_1p5": 0.027777777777777776
      },
      "mean_output_rv_by_condition": {
        "bridge_only_2": 0.6713644001897968,
        "bridge_only_3": 0.6430804655506567,
        "control": 0.657076428282222,
        "early_mlp_0p25_bridge_2": 0.6024387752185033,
        "early_mlp_0p25_bridge_3": 0.6242735004203513,
        "early_mlp_0p5_bridge_2": 0.6592520555554989,
        "early_mlp_0p5_bridge_3": 0.6551452281466965,
        "early_mlp_1_bridge_2": 0.6607812157526188,
        "early_mlp_1_bridge_3": 0.6903761876862232,
        "early_mlp_1p5_bridge_2": 0.7353149452080897,
        "early_mlp_1p5_bridge_3": 0.7430517005508713,
        "early_mlp_only_0p25": 0.645876717674815,
        "early_mlp_only_0p5": 0.6321932357294883,
        "early_mlp_only_1": 0.7077699363220037,
        "early_mlp_only_1p5": 0.7830726736613575
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
      "early_mlp_0p25_bridge_2": {
        "alpha": 2.25,
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
        "rv_cohens_dz": -0.11996169193112215,
        "rv_delta_ci_95": [
          -0.06112837382921254,
          0.03563691666795822
        ],
        "rv_delta_mean": -0.010604506971403346,
        "rv_p_value": 0.6857233667379202
      },
      "early_mlp_0p25_bridge_3": {
        "alpha": 3.25,
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
        "rv_cohens_dz": -0.04298084971122842,
        "rv_delta_ci_95": [
          -0.062028334712932176,
          0.05945819394033126
        ],
        "rv_delta_mean": -0.004900871912901991,
        "rv_p_value": 0.8843349297287685
      },
      "early_mlp_0p5_bridge_2": {
        "alpha": 2.5,
        "bt_art_cohens_h": 0.20379156924782732,
        "bt_art_exact_sign_p": 0.625,
        "bt_art_prompt_losses": 1,
        "bt_art_prompt_wins": 3,
        "bt_art_rate_control": 0.05555555555555555,
        "bt_art_rate_delta": 0.05555555555555555,
        "bt_art_rate_delta_ci_95": [
          -0.05555555555555555,
          0.16666666666666666
        ],
        "bt_art_rate_treated": 0.1111111111111111,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": 0.2501538864927852,
        "rv_delta_ci_95": [
          -0.02770461188297618,
          0.0848475078391675
        ],
        "rv_delta_mean": 0.02617457705356176,
        "rv_p_value": 0.4046896902779651
      },
      "early_mlp_0p5_bridge_3": {
        "alpha": 3.5,
        "bt_art_cohens_h": 0.28790424673741494,
        "bt_art_exact_sign_p": 0.25,
        "bt_art_prompt_losses": 0,
        "bt_art_prompt_wins": 3,
        "bt_art_rate_control": 0.05555555555555555,
        "bt_art_rate_delta": 0.08333333333333333,
        "bt_art_rate_delta_ci_95": [
          0.0,
          0.16666666666666666
        ],
        "bt_art_rate_treated": 0.13888888888888887,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": 0.5166415709739163,
        "rv_delta_ci_95": [
          0.0003626244797844975,
          0.0941305232600547
        ],
        "rv_delta_mean": 0.0443952397225483,
        "rv_p_value": 0.10103640157455071
      },
      "early_mlp_1_bridge_2": {
        "alpha": 3.0,
        "bt_art_cohens_h": 0.10980329379673442,
        "bt_art_exact_sign_p": 1.0,
        "bt_art_prompt_losses": 1,
        "bt_art_prompt_wins": 2,
        "bt_art_rate_control": 0.05555555555555555,
        "bt_art_rate_delta": 0.027777777777777776,
        "bt_art_rate_delta_ci_95": [
          -0.05555555555555555,
          0.1111111111111111
        ],
        "bt_art_rate_treated": 0.08333333333333333,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": 0.1733682002743232,
        "rv_delta_ci_95": [
          -0.04554542839372331,
          0.09810376243024237
        ],
        "rv_delta_mean": 0.0233277033293021,
        "rv_p_value": 0.5602998334807194
      },
      "early_mlp_1_bridge_3": {
        "alpha": 4.0,
        "bt_art_cohens_h": 0.5058831069182061,
        "bt_art_exact_sign_p": 0.21875,
        "bt_art_prompt_losses": 1,
        "bt_art_prompt_wins": 5,
        "bt_art_rate_control": 0.05555555555555555,
        "bt_art_rate_delta": 0.16666666666666666,
        "bt_art_rate_delta_ci_95": [
          0.0,
          0.3333333333333333
        ],
        "bt_art_rate_treated": 0.2222222222222222,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": 0.32578160794151984,
        "rv_delta_ci_95": [
          -0.02304253179729921,
          0.09595417380679658
        ],
        "rv_delta_mean": 0.0359871823936459,
        "rv_p_value": 0.2831037273956761
      },
      "early_mlp_1p5_bridge_2": {
        "alpha": 3.5,
        "bt_art_cohens_h": -0.1409860912210379,
        "bt_art_exact_sign_p": 1.0,
        "bt_art_prompt_losses": 2,
        "bt_art_prompt_wins": 1,
        "bt_art_rate_control": 0.05555555555555555,
        "bt_art_rate_delta": -0.027777777777777776,
        "bt_art_rate_delta_ci_95": [
          -0.1111111111111111,
          0.05555555555555556
        ],
        "bt_art_rate_treated": 0.027777777777777776,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": 0.31098490984857385,
        "rv_delta_ci_95": [
          -0.02089756883962898,
          0.08053876620728012
        ],
        "rv_delta_mean": 0.028501733636507386,
        "rv_p_value": 0.30441211140506647
      },
      "early_mlp_1p5_bridge_3": {
        "alpha": 4.5,
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
        "rv_cohens_dz": 0.6865820058206336,
        "rv_delta_ci_95": [
          0.02299103777336353,
          0.1814662731010075
        ],
        "rv_delta_mean": 0.10088578552870842,
        "rv_p_value": 0.036605134977248446
      },
      "early_mlp_only_0p25": {
        "alpha": 0.25,
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
        "rv_cohens_dz": 0.5100997067445111,
        "rv_delta_ci_95": [
          -0.0004932995085591262,
          0.12764287162809868
        ],
        "rv_delta_mean": 0.06256139283962825,
        "rv_p_value": 0.1049169361622353
      },
      "early_mlp_only_0p5": {
        "alpha": 0.5,
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
        "rv_cohens_dz": 0.3437142587904292,
        "rv_delta_ci_95": [
          -0.02327535486767315,
          0.0732389765221306
        ],
        "rv_delta_mean": 0.029210974794192306,
        "rv_p_value": 0.2588469971368878
      },
      "early_mlp_only_1": {
        "alpha": 1.0,
        "bt_art_cohens_h": -0.1409860912210379,
        "bt_art_exact_sign_p": 1.0,
        "bt_art_prompt_losses": 1,
        "bt_art_prompt_wins": 0,
        "bt_art_rate_control": 0.05555555555555555,
        "bt_art_rate_delta": -0.027777777777777776,
        "bt_art_rate_delta_ci_95": [
          -0.08333333333333333,
          0.0
        ],
        "bt_art_rate_treated": 0.027777777777777776,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": 0.2712909076285275,
        "rv_delta_ci_95": [
          -0.022348762696658014,
          0.07034207822587157
        ],
        "rv_delta_mean": 0.02330606415919002,
        "rv_p_value": 0.36750958902285946
      },
      "early_mlp_only_1p5": {
        "alpha": 1.5,
        "bt_art_cohens_h": -0.4758190041072026,
        "bt_art_exact_sign_p": 0.5,
        "bt_art_prompt_losses": 2,
        "bt_art_prompt_wins": 0,
        "bt_art_rate_control": 0.05555555555555555,
        "bt_art_rate_delta": -0.05555555555555555,
        "bt_art_rate_delta_ci_95": [
          -0.13888888888888887,
          0.0
        ],
        "bt_art_rate_treated": 0.0,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": 0.7921996741127101,
        "rv_delta_ci_95": [
          0.03565294599937969,
          0.16848063995267173
        ],
        "rv_delta_mean": 0.09896847103969364,
        "rv_p_value": 0.019085522079212026
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
      "early_mlp_0p25_bridge_2": {
        "alpha": 2.25,
        "bt_art_cohens_h": 0.2758374268425864,
        "bt_art_exact_sign_p": 0.2890625,
        "bt_art_prompt_losses": 2,
        "bt_art_prompt_wins": 6,
        "bt_art_rate_control": 0.1527777777777778,
        "bt_art_rate_delta": 0.11111111111111109,
        "bt_art_rate_delta_ci_95": [
          -4.625929269271485e-18,
          0.23611111111111108
        ],
        "bt_art_rate_treated": 0.2638888888888889,
        "n_prompt_pairs": 24,
        "rv_cohens_dz": -0.3051993323846919,
        "rv_delta_ci_95": [
          -0.07650501494002712,
          0.005907272531282463
        ],
        "rv_delta_mean": -0.032621080017561054,
        "rv_p_value": 0.14846672263612623
      },
      "early_mlp_0p25_bridge_3": {
        "alpha": 3.25,
        "bt_art_cohens_h": 0.21166536888748866,
        "bt_art_exact_sign_p": 0.34375,
        "bt_art_prompt_losses": 3,
        "bt_art_prompt_wins": 7,
        "bt_art_rate_control": 0.1527777777777778,
        "bt_art_rate_delta": 0.08333333333333333,
        "bt_art_rate_delta_ci_95": [
          -0.013888888888888897,
          0.19444444444444442
        ],
        "bt_art_rate_treated": 0.23611111111111108,
        "n_prompt_pairs": 24,
        "rv_cohens_dz": -0.18375134446494254,
        "rv_delta_ci_95": [
          -0.05842553633040816,
          0.022007777931641766
        ],
        "rv_delta_mean": -0.01885189988738628,
        "rv_p_value": 0.3773462194995807
      },
      "early_mlp_0p5_bridge_2": {
        "alpha": 2.5,
        "bt_art_cohens_h": 0.178616551191757,
        "bt_art_exact_sign_p": 0.2890625,
        "bt_art_prompt_losses": 2,
        "bt_art_prompt_wins": 6,
        "bt_art_rate_control": 0.1527777777777778,
        "bt_art_rate_delta": 0.06944444444444443,
        "bt_art_rate_delta_ci_95": [
          -0.01388888888888889,
          0.15277777777777776
        ],
        "bt_art_rate_treated": 0.2222222222222222,
        "n_prompt_pairs": 24,
        "rv_cohens_dz": 0.14593835451396522,
        "rv_delta_ci_95": [
          -0.020601818513171066,
          0.05364659775326323
        ],
        "rv_delta_mean": 0.014175102163419323,
        "rv_p_value": 0.48183523201705813
      },
      "early_mlp_0p5_bridge_3": {
        "alpha": 3.5,
        "bt_art_cohens_h": 0.21166536888748866,
        "bt_art_exact_sign_p": 0.0625,
        "bt_art_prompt_losses": 0,
        "bt_art_prompt_wins": 5,
        "bt_art_rate_control": 0.1527777777777778,
        "bt_art_rate_delta": 0.08333333333333333,
        "bt_art_rate_delta_ci_95": [
          0.027777777777777776,
          0.1527777777777778
        ],
        "bt_art_rate_treated": 0.23611111111111108,
        "n_prompt_pairs": 24,
        "rv_cohens_dz": 0.21281178739195353,
        "rv_delta_ci_95": [
          -0.01587731107701385,
          0.06075964994911344
        ],
        "rv_delta_mean": 0.021232019793511383,
        "rv_p_value": 0.3079870399987303
      },
      "early_mlp_1_bridge_2": {
        "alpha": 3.0,
        "bt_art_cohens_h": -0.12347498647862187,
        "bt_art_exact_sign_p": 0.75390625,
        "bt_art_prompt_losses": 6,
        "bt_art_prompt_wins": 4,
        "bt_art_rate_control": 0.1527777777777778,
        "bt_art_rate_delta": -0.04166666666666668,
        "bt_art_rate_delta_ci_95": [
          -0.1388888888888889,
          0.05555555555555555
        ],
        "bt_art_rate_treated": 0.1111111111111111,
        "n_prompt_pairs": 24,
        "rv_cohens_dz": 0.1038383279769367,
        "rv_delta_ci_95": [
          -0.034817087198948514,
          0.06473476526177094
        ],
        "rv_delta_mean": 0.013516245399849462,
        "rv_p_value": 0.6158064727686721
      },
      "early_mlp_1_bridge_3": {
        "alpha": 4.0,
        "bt_art_cohens_h": 0.14482093599602797,
        "bt_art_exact_sign_p": 0.5810546875,
        "bt_art_prompt_losses": 5,
        "bt_art_prompt_wins": 8,
        "bt_art_rate_control": 0.1527777777777778,
        "bt_art_rate_delta": 0.055555555555555546,
        "bt_art_rate_delta_ci_95": [
          -0.06944444444444445,
          0.18055555555555555
        ],
        "bt_art_rate_treated": 0.20833333333333334,
        "n_prompt_pairs": 24,
        "rv_cohens_dz": 0.25267198793035783,
        "rv_delta_ci_95": [
          -0.016195859441209208,
          0.08941026852743955
        ],
        "rv_delta_mean": 0.0346434708988236,
        "rv_p_value": 0.22826488424898136
      },
      "early_mlp_1p5_bridge_2": {
        "alpha": 3.5,
        "bt_art_cohens_h": -0.392010943064518,
        "bt_art_exact_sign_p": 0.109375,
        "bt_art_prompt_losses": 8,
        "bt_art_prompt_wins": 2,
        "bt_art_rate_control": 0.1527777777777778,
        "bt_art_rate_delta": -0.1111111111111111,
        "bt_art_rate_delta_ci_95": [
          -0.20833333333333334,
          -0.013888888888888888
        ],
        "bt_art_rate_treated": 0.041666666666666664,
        "n_prompt_pairs": 24,
        "rv_cohens_dz": 0.5059023600304318,
        "rv_delta_ci_95": [
          0.012690342854282572,
          0.09431105790504689
        ],
        "rv_delta_mean": 0.05337012528118751,
        "rv_p_value": 0.020967834060091745
      },
      "early_mlp_1p5_bridge_3": {
        "alpha": 4.5,
        "bt_art_cohens_h": -0.16896496456282495,
        "bt_art_exact_sign_p": 0.548828125,
        "bt_art_prompt_losses": 7,
        "bt_art_prompt_wins": 4,
        "bt_art_rate_control": 0.1527777777777778,
        "bt_art_rate_delta": -0.05555555555555555,
        "bt_art_rate_delta_ci_95": [
          -0.18055555555555555,
          0.06944444444444443
        ],
        "bt_art_rate_treated": 0.09722222222222221,
        "n_prompt_pairs": 24,
        "rv_cohens_dz": 0.6486658566433753,
        "rv_delta_ci_95": [
          0.038887035626436556,
          0.14963659792975775
        ],
        "rv_delta_mean": 0.09343052889867887,
        "rv_p_value": 0.0041952138055366996
      },
      "early_mlp_only_0p25": {
        "alpha": 0.25,
        "bt_art_cohens_h": 0.24404874580973213,
        "bt_art_exact_sign_p": 0.109375,
        "bt_art_prompt_losses": 2,
        "bt_art_prompt_wins": 8,
        "bt_art_rate_control": 0.1527777777777778,
        "bt_art_rate_delta": 0.09722222222222221,
        "bt_art_rate_delta_ci_95": [
          0.01388888888888888,
          0.19444444444444442
        ],
        "bt_art_rate_treated": 0.25,
        "n_prompt_pairs": 24,
        "rv_cohens_dz": 0.20235205815771865,
        "rv_delta_ci_95": [
          -0.024377946663245987,
          0.07345610199150522
        ],
        "rv_delta_mean": 0.025680841116110704,
        "rv_p_value": 0.33184525659074615
      },
      "early_mlp_only_0p5": {
        "alpha": 0.5,
        "bt_art_cohens_h": 0.07459443982805836,
        "bt_art_exact_sign_p": 1.0,
        "bt_art_prompt_losses": 6,
        "bt_art_prompt_wins": 6,
        "bt_art_rate_control": 0.1527777777777778,
        "bt_art_rate_delta": 0.027777777777777776,
        "bt_art_rate_delta_ci_95": [
          -0.08333333333333333,
          0.13888888888888892
        ],
        "bt_art_rate_treated": 0.18055555555555555,
        "n_prompt_pairs": 24,
        "rv_cohens_dz": 0.024433576269870437,
        "rv_delta_ci_95": [
          -0.03202132529748981,
          0.034628124062652255
        ],
        "rv_delta_mean": 0.002163891120729291,
        "rv_p_value": 0.9057605579599033
      },
      "early_mlp_only_1": {
        "alpha": 1.0,
        "bt_art_cohens_h": -0.4682526469474871,
        "bt_art_exact_sign_p": 0.015625,
        "bt_art_prompt_losses": 7,
        "bt_art_prompt_wins": 0,
        "bt_art_rate_control": 0.1527777777777778,
        "bt_art_rate_delta": -0.125,
        "bt_art_rate_delta_ci_95": [
          -0.2222222222222222,
          -0.041666666666666664
        ],
        "bt_art_rate_treated": 0.027777777777777776,
        "n_prompt_pairs": 24,
        "rv_cohens_dz": 0.35438383998282524,
        "rv_delta_ci_95": [
          -0.004974738947237124,
          0.07635332569998718
        ],
        "rv_delta_mean": 0.03699978609948585,
        "rv_p_value": 0.0959218204186907
      },
      "early_mlp_only_1p5": {
        "alpha": 1.5,
        "bt_art_cohens_h": -0.5668974994020952,
        "bt_art_exact_sign_p": 0.0078125,
        "bt_art_prompt_losses": 8,
        "bt_art_prompt_wins": 0,
        "bt_art_rate_control": 0.1527777777777778,
        "bt_art_rate_delta": -0.13888888888888887,
        "bt_art_rate_delta_ci_95": [
          -0.23611111111111108,
          -0.06944444444444443
        ],
        "bt_art_rate_treated": 0.013888888888888888,
        "n_prompt_pairs": 24,
        "rv_cohens_dz": 0.7373476951055787,
        "rv_delta_ci_95": [
          0.053213257170601345,
          0.17274839396525138
        ],
        "rv_delta_mean": 0.11248235820941449,
        "rv_p_value": 0.0014653880232861926
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
      "early_mlp_0p25_bridge_2": {
        "alpha": 2.25,
        "bt_art_cohens_h": 0.3561506963786092,
        "bt_art_exact_sign_p": 0.375,
        "bt_art_prompt_losses": 1,
        "bt_art_prompt_wins": 4,
        "bt_art_rate_control": 0.25,
        "bt_art_rate_delta": 0.16666666666666666,
        "bt_art_rate_delta_ci_95": [
          -9.367506770274719e-18,
          0.36111111111111116
        ],
        "bt_art_rate_treated": 0.4166666666666666,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": -0.44588910354500644,
        "rv_delta_ci_95": [
          -0.1269902133995787,
          0.007589508086091342
        ],
        "rv_delta_mean": -0.054637653063718754,
        "rv_p_value": 0.1507082440337898
      },
      "early_mlp_0p25_bridge_3": {
        "alpha": 3.25,
        "bt_art_cohens_h": 0.29950568329692784,
        "bt_art_exact_sign_p": 0.2890625,
        "bt_art_prompt_losses": 2,
        "bt_art_prompt_wins": 6,
        "bt_art_rate_control": 0.25,
        "bt_art_rate_delta": 0.13888888888888887,
        "bt_art_rate_delta_ci_95": [
          -0.027777777777777787,
          0.3055555555555556
        ],
        "bt_art_rate_treated": 0.38888888888888884,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": -0.35409498036988607,
        "rv_delta_ci_95": [
          -0.08537574399834666,
          0.01817932379982983
        ],
        "rv_delta_mean": -0.03280292786187058,
        "rv_p_value": 0.2455732479392418
      },
      "early_mlp_0p5_bridge_2": {
        "alpha": 2.5,
        "bt_art_cohens_h": 0.18376186614417667,
        "bt_art_exact_sign_p": 0.625,
        "bt_art_prompt_losses": 1,
        "bt_art_prompt_wins": 3,
        "bt_art_rate_control": 0.25,
        "bt_art_rate_delta": 0.08333333333333333,
        "bt_art_rate_delta_ci_95": [
          -0.05555555555555556,
          0.2222222222222222
        ],
        "bt_art_rate_treated": 0.3333333333333333,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": 0.023648081100992802,
        "rv_delta_ci_95": [
          -0.04692691145747714,
          0.053865001316091295
        ],
        "rv_delta_mean": 0.002175627273276882,
        "rv_p_value": 0.9361824422789862
      },
      "early_mlp_0p5_bridge_3": {
        "alpha": 3.5,
        "bt_art_cohens_h": 0.18376186614417667,
        "bt_art_exact_sign_p": 0.5,
        "bt_art_prompt_losses": 0,
        "bt_art_prompt_wins": 2,
        "bt_art_rate_control": 0.25,
        "bt_art_rate_delta": 0.08333333333333333,
        "bt_art_rate_delta_ci_95": [
          0.0,
          0.19444444444444445
        ],
        "bt_art_rate_treated": 0.3333333333333333,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": -0.017442893076813383,
        "rv_delta_ci_95": [
          -0.061330772845699585,
          0.05942947666257818
        ],
        "rv_delta_mean": -0.0019312001355255338,
        "rv_p_value": 0.9529018177108066
      },
      "early_mlp_1_bridge_2": {
        "alpha": 3.0,
        "bt_art_cohens_h": -0.2834110547987664,
        "bt_art_exact_sign_p": 0.453125,
        "bt_art_prompt_losses": 5,
        "bt_art_prompt_wins": 2,
        "bt_art_rate_control": 0.25,
        "bt_art_rate_delta": -0.1111111111111111,
        "bt_art_rate_delta_ci_95": [
          -0.27777777777777773,
          0.055555555555555546
        ],
        "bt_art_rate_treated": 0.13888888888888887,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": 0.028321701330648204,
        "rv_delta_ci_95": [
          -0.06794734950611815,
          0.0715450617166322
        ],
        "rv_delta_mean": 0.0037047874703968245,
        "rv_p_value": 0.9236104696676493
      },
      "early_mlp_1_bridge_3": {
        "alpha": 4.0,
        "bt_art_cohens_h": -0.13386478078334507,
        "bt_art_exact_sign_p": 1.0,
        "bt_art_prompt_losses": 4,
        "bt_art_prompt_wins": 3,
        "bt_art_rate_control": 0.25,
        "bt_art_rate_delta": -0.05555555555555556,
        "bt_art_rate_delta_ci_95": [
          -0.22222222222222224,
          0.1111111111111111
        ],
        "bt_art_rate_treated": 0.19444444444444442,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": 0.20228096165856718,
        "rv_delta_ci_95": [
          -0.05265596522070019,
          0.12697200856737087
        ],
        "rv_delta_mean": 0.033299759404001296,
        "rv_p_value": 0.49803112946400796
      },
      "early_mlp_1p5_bridge_2": {
        "alpha": 3.5,
        "bt_art_cohens_h": -0.5713153015361814,
        "bt_art_exact_sign_p": 0.125,
        "bt_art_prompt_losses": 6,
        "bt_art_prompt_wins": 1,
        "bt_art_rate_control": 0.25,
        "bt_art_rate_delta": -0.19444444444444445,
        "bt_art_rate_delta_ci_95": [
          -0.3611111111111111,
          -0.027777777777777787
        ],
        "bt_art_rate_treated": 0.05555555555555555,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": 0.6728449074583689,
        "rv_delta_ci_95": [
          0.015390079694045308,
          0.1409941241225431
        ],
        "rv_delta_mean": 0.07823851692586761,
        "rv_p_value": 0.039812673602498246
      },
      "early_mlp_1p5_bridge_3": {
        "alpha": 4.5,
        "bt_art_cohens_h": -0.20612888062866763,
        "bt_art_exact_sign_p": 0.7265625,
        "bt_art_prompt_losses": 5,
        "bt_art_prompt_wins": 3,
        "bt_art_rate_control": 0.25,
        "bt_art_rate_delta": -0.08333333333333333,
        "bt_art_rate_delta_ci_95": [
          -0.3055555555555556,
          0.1388888888888889
        ],
        "bt_art_rate_treated": 0.16666666666666666,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": 0.5840995679397823,
        "rv_delta_ci_95": [
          0.0017168627560824568,
          0.16299304454231092
        ],
        "rv_delta_mean": 0.08597527226864932,
        "rv_p_value": 0.0680202150163469
      },
      "early_mlp_only_0p25": {
        "alpha": 0.25,
        "bt_art_cohens_h": 0.29950568329692784,
        "bt_art_exact_sign_p": 0.2890625,
        "bt_art_prompt_losses": 2,
        "bt_art_prompt_wins": 6,
        "bt_art_rate_control": 0.25,
        "bt_art_rate_delta": 0.13888888888888887,
        "bt_art_rate_delta_ci_95": [
          -0.027777777777777787,
          0.3055555555555555
        ],
        "bt_art_rate_treated": 0.3888888888888889,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": -0.08947512889570362,
        "rv_delta_ci_95": [
          -0.08470222655259177,
          0.05172925865590353
        ],
        "rv_delta_mean": -0.011199710607406835,
        "rv_p_value": 0.7623912271685608
      },
      "early_mlp_only_0p5": {
        "alpha": 0.5,
        "bt_art_cohens_h": 0.12417353571770429,
        "bt_art_exact_sign_p": 1.0,
        "bt_art_prompt_losses": 4,
        "bt_art_prompt_wins": 4,
        "bt_art_rate_control": 0.25,
        "bt_art_rate_delta": 0.05555555555555556,
        "bt_art_rate_delta_ci_95": [
          -0.13888888888888887,
          0.25
        ],
        "bt_art_rate_treated": 0.3055555555555556,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": -0.2857911450457183,
        "rv_delta_ci_95": [
          -0.07252699956969567,
          0.021340554957521476
        ],
        "rv_delta_mean": -0.024883192552733722,
        "rv_p_value": 0.34344693249964897
      },
      "early_mlp_only_1": {
        "alpha": 1.0,
        "bt_art_cohens_h": -0.7123013927572193,
        "bt_art_exact_sign_p": 0.03125,
        "bt_art_prompt_losses": 6,
        "bt_art_prompt_wins": 0,
        "bt_art_rate_control": 0.25,
        "bt_art_rate_delta": -0.22222222222222224,
        "bt_art_rate_delta_ci_95": [
          -0.3611111111111111,
          -0.08333333333333333
        ],
        "bt_art_rate_treated": 0.027777777777777776,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": 0.41387135341582554,
        "rv_delta_ci_95": [
          -0.019822742454550332,
          0.11365522719868877
        ],
        "rv_delta_mean": 0.050693508039781676,
        "rv_p_value": 0.17946795608074073
      },
      "early_mlp_only_1p5": {
        "alpha": 1.5,
        "bt_art_cohens_h": -0.7123013927572193,
        "bt_art_exact_sign_p": 0.03125,
        "bt_art_prompt_losses": 6,
        "bt_art_prompt_wins": 0,
        "bt_art_rate_control": 0.25,
        "bt_art_rate_delta": -0.22222222222222224,
        "bt_art_rate_delta_ci_95": [
          -0.36111111111111116,
          -0.08333333333333333
        ],
        "bt_art_rate_treated": 0.027777777777777776,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": 0.6972628012175665,
        "rv_delta_ci_95": [
          0.02809309369405282,
          0.22554645921032282
        ],
        "rv_delta_mean": 0.12599624537913534,
        "rv_p_value": 0.03428572362009206
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
        "bridge": 0.0,
        "early_mlp": 0.25
      },
      "name": "early_mlp_only_0p25"
    },
    {
      "alphas": {
        "bridge": 0.0,
        "early_mlp": 0.5
      },
      "name": "early_mlp_only_0p5"
    },
    {
      "alphas": {
        "bridge": 0.0,
        "early_mlp": 1.0
      },
      "name": "early_mlp_only_1"
    },
    {
      "alphas": {
        "bridge": 0.0,
        "early_mlp": 1.5
      },
      "name": "early_mlp_only_1p5"
    },
    {
      "alphas": {
        "bridge": 2.0,
        "early_mlp": 0.25
      },
      "name": "early_mlp_0p25_bridge_2"
    },
    {
      "alphas": {
        "bridge": 2.0,
        "early_mlp": 0.5
      },
      "name": "early_mlp_0p5_bridge_2"
    },
    {
      "alphas": {
        "bridge": 2.0,
        "early_mlp": 1.0
      },
      "name": "early_mlp_1_bridge_2"
    },
    {
      "alphas": {
        "bridge": 2.0,
        "early_mlp": 1.5
      },
      "name": "early_mlp_1p5_bridge_2"
    },
    {
      "alphas": {
        "bridge": 3.0,
        "early_mlp": 0.25
      },
      "name": "early_mlp_0p25_bridge_3"
    },
    {
      "alphas": {
        "bridge": 3.0,
        "early_mlp": 0.5
      },
      "name": "early_mlp_0p5_bridge_3"
    },
    {
      "alphas": {
        "bridge": 3.0,
        "early_mlp": 1.0
      },
      "name": "early_mlp_1_bridge_3"
    },
    {
      "alphas": {
        "bridge": 3.0,
        "early_mlp": 1.5
      },
      "name": "early_mlp_1p5_bridge_3"
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
      "centroid_cosine": 0.7863364219665527,
      "component": "mlp",
      "direction_norm": 0.08519649505615234,
      "layer": 4,
      "token_window": 16,
      "window": 16
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
  "timestamp": "20260312_100649",
  "top_p": 0.95,
  "verdict": "multisite_sufficient"
}
```
