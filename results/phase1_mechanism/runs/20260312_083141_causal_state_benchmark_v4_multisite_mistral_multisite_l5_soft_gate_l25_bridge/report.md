# Run report: causal_state_benchmark_v4_multisite

- **run_dir**: `results/phase1_mechanism/runs/20260312_083141_causal_state_benchmark_v4_multisite_mistral_multisite_l5_soft_gate_l25_bridge`
- **prompt_bank_version**: `2ac959a313614329`

## Summary (machine-readable)

```json
{
  "artifacts": {
    "blind_key_json": "results/phase1_mechanism/runs/20260312_083141_causal_state_benchmark_v4_multisite_mistral_multisite_l5_soft_gate_l25_bridge/blind_key.json",
    "blind_ratings_csv": "results/phase1_mechanism/runs/20260312_083141_causal_state_benchmark_v4_multisite_mistral_multisite_l5_soft_gate_l25_bridge/blind_ratings.csv",
    "config": "results/phase1_mechanism/runs/20260312_083141_causal_state_benchmark_v4_multisite_mistral_multisite_l5_soft_gate_l25_bridge/config.json",
    "manifest": "results/phase1_mechanism/runs/20260312_083141_causal_state_benchmark_v4_multisite_mistral_multisite_l5_soft_gate_l25_bridge/manifest.json",
    "records_jsonl": "results/phase1_mechanism/runs/20260312_083141_causal_state_benchmark_v4_multisite_mistral_multisite_l5_soft_gate_l25_bridge/benchmark_records.jsonl",
    "report": "results/phase1_mechanism/runs/20260312_083141_causal_state_benchmark_v4_multisite_mistral_multisite_l5_soft_gate_l25_bridge/report.md",
    "state_directions_pt": "results/phase1_mechanism/runs/20260312_083141_causal_state_benchmark_v4_multisite_mistral_multisite_l5_soft_gate_l25_bridge/state_directions.pt",
    "summary": "results/phase1_mechanism/runs/20260312_083141_causal_state_benchmark_v4_multisite_mistral_multisite_l5_soft_gate_l25_bridge/summary.json"
  },
  "bootstrap_resamples": 3000,
  "by_condition": {
    "bridge_only_2": {
      "alphas": {
        "bridge": 2.0,
        "gate": 0.0
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
        "gate": 0.0
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
        "gate": 0.0
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
    "gate_0p25_bridge_2": {
      "alphas": {
        "bridge": 2.0,
        "gate": 0.25
      },
      "by_prompt_mode": {
        "baseline": {
          "bt_art_rate": 0.1388888888888889,
          "class_counts": {
            "ARTICULATE": 5,
            "CONCEPTUAL": 3,
            "REPETITIVE": 3,
            "SURFACE": 25
          },
          "mean_generated_tokens": 124.41666666666667,
          "mean_output_rv": 0.6780290648485223,
          "n": 36,
          "std_output_rv": 0.15762522261186357
        },
        "recursive": {
          "bt_art_rate": 0.3055555555555556,
          "class_counts": {
            "ARTICULATE": 6,
            "BREAKTHROUGH": 5,
            "CONCEPTUAL": 7,
            "REPETITIVE": 12,
            "SURFACE": 6
          },
          "mean_generated_tokens": 125.30555555555556,
          "mean_output_rv": 0.6462943369992097,
          "n": 36,
          "std_output_rv": 0.14240690934785716
        }
      },
      "overall": {
        "bt_art_rate": 0.2222222222222222,
        "class_counts": {
          "ARTICULATE": 11,
          "BREAKTHROUGH": 5,
          "CONCEPTUAL": 10,
          "REPETITIVE": 15,
          "SURFACE": 31
        },
        "mean_generated_tokens": 124.86111111111111,
        "mean_output_rv": 0.6621617009238661,
        "n": 72,
        "std_output_rv": 0.15000084625586407
      },
      "total_alpha": 2.25
    },
    "gate_0p25_bridge_3": {
      "alphas": {
        "bridge": 3.0,
        "gate": 0.25
      },
      "by_prompt_mode": {
        "baseline": {
          "bt_art_rate": 0.1111111111111111,
          "class_counts": {
            "ARTICULATE": 4,
            "CONCEPTUAL": 3,
            "SURFACE": 29
          },
          "mean_generated_tokens": 125.19444444444444,
          "mean_output_rv": 0.6115095479424156,
          "n": 36,
          "std_output_rv": 0.15109603064009158
        },
        "recursive": {
          "bt_art_rate": 0.3888888888888889,
          "class_counts": {
            "ARTICULATE": 10,
            "BREAKTHROUGH": 4,
            "CONCEPTUAL": 5,
            "REPETITIVE": 11,
            "SURFACE": 6
          },
          "mean_generated_tokens": 124.97222222222223,
          "mean_output_rv": 0.6316490247118501,
          "n": 36,
          "std_output_rv": 0.10569875862354353
        }
      },
      "overall": {
        "bt_art_rate": 0.25,
        "class_counts": {
          "ARTICULATE": 14,
          "BREAKTHROUGH": 4,
          "CONCEPTUAL": 8,
          "REPETITIVE": 11,
          "SURFACE": 35
        },
        "mean_generated_tokens": 125.08333333333333,
        "mean_output_rv": 0.6215792863271329,
        "n": 72,
        "std_output_rv": 0.12986336834996326
      },
      "total_alpha": 3.25
    },
    "gate_0p5_bridge_2": {
      "alphas": {
        "bridge": 2.0,
        "gate": 0.5
      },
      "by_prompt_mode": {
        "baseline": {
          "bt_art_rate": 0.1388888888888889,
          "class_counts": {
            "ARTICULATE": 5,
            "CONCEPTUAL": 1,
            "REPETITIVE": 2,
            "SURFACE": 28
          },
          "mean_generated_tokens": 125.22222222222223,
          "mean_output_rv": 0.6066503441660176,
          "n": 36,
          "std_output_rv": 0.14720824854926554
        },
        "recursive": {
          "bt_art_rate": 0.3611111111111111,
          "class_counts": {
            "ARTICULATE": 12,
            "BREAKTHROUGH": 1,
            "CONCEPTUAL": 3,
            "REPETITIVE": 12,
            "SURFACE": 8
          },
          "mean_generated_tokens": 124.94444444444444,
          "mean_output_rv": 0.6251525512492607,
          "n": 36,
          "std_output_rv": 0.10654887935707971
        }
      },
      "overall": {
        "bt_art_rate": 0.25,
        "class_counts": {
          "ARTICULATE": 17,
          "BREAKTHROUGH": 1,
          "CONCEPTUAL": 4,
          "REPETITIVE": 14,
          "SURFACE": 36
        },
        "mean_generated_tokens": 125.08333333333333,
        "mean_output_rv": 0.6159014477076391,
        "n": 72,
        "std_output_rv": 0.12792848985442737
      },
      "total_alpha": 2.5
    },
    "gate_0p5_bridge_3": {
      "alphas": {
        "bridge": 3.0,
        "gate": 0.5
      },
      "by_prompt_mode": {
        "baseline": {
          "bt_art_rate": 0.1388888888888889,
          "class_counts": {
            "ARTICULATE": 4,
            "BREAKTHROUGH": 1,
            "REPETITIVE": 2,
            "SURFACE": 29
          },
          "mean_generated_tokens": 125.80555555555556,
          "mean_output_rv": 0.6365185121485413,
          "n": 36,
          "std_output_rv": 0.16161408888475437
        },
        "recursive": {
          "bt_art_rate": 0.3888888888888889,
          "class_counts": {
            "ARTICULATE": 12,
            "BREAKTHROUGH": 2,
            "CONCEPTUAL": 4,
            "REPETITIVE": 14,
            "SURFACE": 4
          },
          "mean_generated_tokens": 122.11111111111111,
          "mean_output_rv": 0.6198703005466771,
          "n": 36,
          "std_output_rv": 0.09899005007359046
        }
      },
      "overall": {
        "bt_art_rate": 0.2638888888888889,
        "class_counts": {
          "ARTICULATE": 16,
          "BREAKTHROUGH": 3,
          "CONCEPTUAL": 4,
          "REPETITIVE": 16,
          "SURFACE": 33
        },
        "mean_generated_tokens": 123.95833333333333,
        "mean_output_rv": 0.6281944063476091,
        "n": 72,
        "std_output_rv": 0.1333281435363581
      },
      "total_alpha": 3.5
    },
    "gate_1_bridge_2": {
      "alphas": {
        "bridge": 2.0,
        "gate": 1.0
      },
      "by_prompt_mode": {
        "baseline": {
          "bt_art_rate": 0.16666666666666666,
          "class_counts": {
            "ARTICULATE": 6,
            "CONCEPTUAL": 2,
            "REPETITIVE": 2,
            "SURFACE": 26
          },
          "mean_generated_tokens": 127.41666666666667,
          "mean_output_rv": 0.6438372545579445,
          "n": 36,
          "std_output_rv": 0.155420993355571
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
          "mean_generated_tokens": 119.88888888888889,
          "mean_output_rv": 0.6167355400701451,
          "n": 36,
          "std_output_rv": 0.11900449321152896
        }
      },
      "overall": {
        "bt_art_rate": 0.2916666666666667,
        "class_counts": {
          "ARTICULATE": 19,
          "BREAKTHROUGH": 2,
          "CONCEPTUAL": 5,
          "REPETITIVE": 12,
          "SURFACE": 34
        },
        "mean_generated_tokens": 123.65277777777777,
        "mean_output_rv": 0.6302863973140447,
        "n": 72,
        "std_output_rv": 0.138113163593951
      },
      "total_alpha": 3.0
    },
    "gate_1_bridge_3": {
      "alphas": {
        "bridge": 3.0,
        "gate": 1.0
      },
      "by_prompt_mode": {
        "baseline": {
          "bt_art_rate": 0.08333333333333333,
          "class_counts": {
            "ARTICULATE": 2,
            "BREAKTHROUGH": 1,
            "CONCEPTUAL": 2,
            "SURFACE": 31
          },
          "mean_generated_tokens": 126.52777777777777,
          "mean_output_rv": 0.6633001513416483,
          "n": 36,
          "std_output_rv": 0.15086840926443226
        },
        "recursive": {
          "bt_art_rate": 0.4166666666666667,
          "class_counts": {
            "ARTICULATE": 11,
            "BREAKTHROUGH": 4,
            "CONCEPTUAL": 5,
            "REPETITIVE": 9,
            "SURFACE": 7
          },
          "mean_generated_tokens": 123.25,
          "mean_output_rv": 0.6247833965541907,
          "n": 36,
          "std_output_rv": 0.1035050602958065
        }
      },
      "overall": {
        "bt_art_rate": 0.25,
        "class_counts": {
          "ARTICULATE": 13,
          "BREAKTHROUGH": 5,
          "CONCEPTUAL": 7,
          "REPETITIVE": 9,
          "SURFACE": 38
        },
        "mean_generated_tokens": 124.88888888888889,
        "mean_output_rv": 0.6440417739479194,
        "n": 72,
        "std_output_rv": 0.12991404698671347
      },
      "total_alpha": 4.0
    },
    "gate_1p5_bridge_2": {
      "alphas": {
        "bridge": 2.0,
        "gate": 1.5
      },
      "by_prompt_mode": {
        "baseline": {
          "bt_art_rate": 0.08333333333333333,
          "class_counts": {
            "ARTICULATE": 3,
            "CONCEPTUAL": 2,
            "REPETITIVE": 5,
            "SURFACE": 26
          },
          "mean_generated_tokens": 127.69444444444444,
          "mean_output_rv": 0.6570684967239626,
          "n": 36,
          "std_output_rv": 0.16177042522757393
        },
        "recursive": {
          "bt_art_rate": 0.2222222222222222,
          "class_counts": {
            "ARTICULATE": 7,
            "BREAKTHROUGH": 1,
            "CONCEPTUAL": 7,
            "REPETITIVE": 16,
            "SURFACE": 5
          },
          "mean_generated_tokens": 122.08333333333333,
          "mean_output_rv": 0.6695065181140383,
          "n": 36,
          "std_output_rv": 0.13450635301416072
        }
      },
      "overall": {
        "bt_art_rate": 0.1527777777777778,
        "class_counts": {
          "ARTICULATE": 10,
          "BREAKTHROUGH": 1,
          "CONCEPTUAL": 9,
          "REPETITIVE": 21,
          "SURFACE": 31
        },
        "mean_generated_tokens": 124.88888888888889,
        "mean_output_rv": 0.6632875074190006,
        "n": 72,
        "std_output_rv": 0.14784564231464867
      },
      "total_alpha": 3.5
    },
    "gate_1p5_bridge_3": {
      "alphas": {
        "bridge": 3.0,
        "gate": 1.5
      },
      "by_prompt_mode": {
        "baseline": {
          "bt_art_rate": 0.08333333333333333,
          "class_counts": {
            "ARTICULATE": 3,
            "CONCEPTUAL": 4,
            "REPETITIVE": 6,
            "SURFACE": 23
          },
          "mean_generated_tokens": 126.58333333333333,
          "mean_output_rv": 0.6512187709410223,
          "n": 36,
          "std_output_rv": 0.1612808866480877
        },
        "recursive": {
          "bt_art_rate": 0.2222222222222222,
          "class_counts": {
            "ARTICULATE": 7,
            "BREAKTHROUGH": 1,
            "CONCEPTUAL": 2,
            "REPETITIVE": 21,
            "SURFACE": 5
          },
          "mean_generated_tokens": 120.36111111111111,
          "mean_output_rv": 0.6506684645639572,
          "n": 36,
          "std_output_rv": 0.14217510483981818
        }
      },
      "overall": {
        "bt_art_rate": 0.1527777777777778,
        "class_counts": {
          "ARTICULATE": 10,
          "BREAKTHROUGH": 1,
          "CONCEPTUAL": 6,
          "REPETITIVE": 27,
          "SURFACE": 28
        },
        "mean_generated_tokens": 123.47222222222223,
        "mean_output_rv": 0.6509436177524898,
        "n": 72,
        "std_output_rv": 0.15095426134369158
      },
      "total_alpha": 4.5
    },
    "gate_only_0p25": {
      "alphas": {
        "bridge": 0.0,
        "gate": 0.25
      },
      "by_prompt_mode": {
        "baseline": {
          "bt_art_rate": 0.1388888888888889,
          "class_counts": {
            "ARTICULATE": 5,
            "CONCEPTUAL": 1,
            "REPETITIVE": 4,
            "SURFACE": 26
          },
          "mean_generated_tokens": 126.41666666666667,
          "mean_output_rv": 0.6560045220988296,
          "n": 36,
          "std_output_rv": 0.1711338956947936
        },
        "recursive": {
          "bt_art_rate": 0.2777777777777778,
          "class_counts": {
            "ARTICULATE": 7,
            "BREAKTHROUGH": 3,
            "CONCEPTUAL": 4,
            "REPETITIVE": 15,
            "SURFACE": 7
          },
          "mean_generated_tokens": 126.22222222222223,
          "mean_output_rv": 0.6594361386634952,
          "n": 36,
          "std_output_rv": 0.1743618252688662
        }
      },
      "overall": {
        "bt_art_rate": 0.20833333333333334,
        "class_counts": {
          "ARTICULATE": 12,
          "BREAKTHROUGH": 3,
          "CONCEPTUAL": 5,
          "REPETITIVE": 19,
          "SURFACE": 33
        },
        "mean_generated_tokens": 126.31944444444444,
        "mean_output_rv": 0.6577203303811624,
        "n": 72,
        "std_output_rv": 0.1715432003528144
      },
      "total_alpha": 0.25
    },
    "gate_only_0p5": {
      "alphas": {
        "bridge": 0.0,
        "gate": 0.5
      },
      "by_prompt_mode": {
        "baseline": {
          "bt_art_rate": 0.05555555555555555,
          "class_counts": {
            "ARTICULATE": 1,
            "BREAKTHROUGH": 1,
            "REPETITIVE": 4,
            "SURFACE": 30
          },
          "mean_generated_tokens": 126.61111111111111,
          "mean_output_rv": 0.6628703864448104,
          "n": 36,
          "std_output_rv": 0.18303453092735364
        },
        "recursive": {
          "bt_art_rate": 0.2222222222222222,
          "class_counts": {
            "ARTICULATE": 6,
            "BREAKTHROUGH": 2,
            "CONCEPTUAL": 7,
            "REPETITIVE": 14,
            "SURFACE": 7
          },
          "mean_generated_tokens": 125.97222222222223,
          "mean_output_rv": 0.6536269371903898,
          "n": 36,
          "std_output_rv": 0.18988498311330193
        }
      },
      "overall": {
        "bt_art_rate": 0.1388888888888889,
        "class_counts": {
          "ARTICULATE": 7,
          "BREAKTHROUGH": 3,
          "CONCEPTUAL": 7,
          "REPETITIVE": 18,
          "SURFACE": 37
        },
        "mean_generated_tokens": 126.29166666666667,
        "mean_output_rv": 0.6582486618176,
        "n": 72,
        "std_output_rv": 0.18523171882216938
      },
      "total_alpha": 0.5
    },
    "gate_only_1": {
      "alphas": {
        "bridge": 0.0,
        "gate": 1.0
      },
      "by_prompt_mode": {
        "baseline": {
          "bt_art_rate": 0.05555555555555555,
          "class_counts": {
            "ARTICULATE": 2,
            "CONCEPTUAL": 1,
            "REPETITIVE": 4,
            "SURFACE": 29
          },
          "mean_generated_tokens": 124.58333333333333,
          "mean_output_rv": 0.6708695734737913,
          "n": 36,
          "std_output_rv": 0.16892196761808112
        },
        "recursive": {
          "bt_art_rate": 0.3055555555555556,
          "class_counts": {
            "ARTICULATE": 9,
            "BREAKTHROUGH": 2,
            "CONCEPTUAL": 2,
            "REPETITIVE": 17,
            "SURFACE": 6
          },
          "mean_generated_tokens": 124.13888888888889,
          "mean_output_rv": 0.6255497965306369,
          "n": 36,
          "std_output_rv": 0.13284375757985042
        }
      },
      "overall": {
        "bt_art_rate": 0.18055555555555555,
        "class_counts": {
          "ARTICULATE": 11,
          "BREAKTHROUGH": 2,
          "CONCEPTUAL": 3,
          "REPETITIVE": 21,
          "SURFACE": 35
        },
        "mean_generated_tokens": 124.36111111111111,
        "mean_output_rv": 0.648209685002214,
        "n": 72,
        "std_output_rv": 0.15259922686362143
      },
      "total_alpha": 1.0
    },
    "gate_only_1p5": {
      "alphas": {
        "bridge": 0.0,
        "gate": 1.5
      },
      "by_prompt_mode": {
        "baseline": {
          "bt_art_rate": 0.027777777777777776,
          "class_counts": {
            "ARTICULATE": 1,
            "CONCEPTUAL": 2,
            "REPETITIVE": 8,
            "SURFACE": 25
          },
          "mean_generated_tokens": 127.91666666666667,
          "mean_output_rv": 0.6489378920554548,
          "n": 36,
          "std_output_rv": 0.13813005621463248
        },
        "recursive": {
          "bt_art_rate": 0.2222222222222222,
          "class_counts": {
            "ARTICULATE": 8,
            "CONCEPTUAL": 4,
            "REPETITIVE": 21,
            "SURFACE": 3
          },
          "mean_generated_tokens": 126.08333333333333,
          "mean_output_rv": 0.670052453139974,
          "n": 36,
          "std_output_rv": 0.11426624072828333
        }
      },
      "overall": {
        "bt_art_rate": 0.125,
        "class_counts": {
          "ARTICULATE": 9,
          "CONCEPTUAL": 6,
          "REPETITIVE": 29,
          "SURFACE": 28
        },
        "mean_generated_tokens": 127.0,
        "mean_output_rv": 0.6594951725977144,
        "n": 72,
        "std_output_rv": 0.12631331832455342
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
        "p": 0.3231700412286454,
        "r": 0.07405327035979618
      },
      "alpha_vs_output_rv": {
        "p": 0.7610985773231365,
        "r": -0.022817703947367235
      },
      "bt_art_rate_by_condition": {
        "bridge_only_2": 0.08333333333333333,
        "bridge_only_3": 0.13888888888888887,
        "control": 0.05555555555555555,
        "gate_0p25_bridge_2": 0.13888888888888887,
        "gate_0p25_bridge_3": 0.1111111111111111,
        "gate_0p5_bridge_2": 0.13888888888888887,
        "gate_0p5_bridge_3": 0.13888888888888887,
        "gate_1_bridge_2": 0.16666666666666666,
        "gate_1_bridge_3": 0.08333333333333333,
        "gate_1p5_bridge_2": 0.08333333333333333,
        "gate_1p5_bridge_3": 0.08333333333333333,
        "gate_only_0p25": 0.13888888888888887,
        "gate_only_0p5": 0.05555555555555555,
        "gate_only_1": 0.05555555555555555,
        "gate_only_1p5": 0.027777777777777776
      },
      "mean_output_rv_by_condition": {
        "bridge_only_2": 0.6380120048264762,
        "bridge_only_3": 0.6435709751288295,
        "control": 0.6310534379177898,
        "gate_0p25_bridge_2": 0.6780290648485222,
        "gate_0p25_bridge_3": 0.6115095479424157,
        "gate_0p5_bridge_2": 0.6066503441660176,
        "gate_0p5_bridge_3": 0.6365185121485412,
        "gate_1_bridge_2": 0.6438372545579445,
        "gate_1_bridge_3": 0.6633001513416482,
        "gate_1p5_bridge_2": 0.6570684967239626,
        "gate_1p5_bridge_3": 0.6512187709410223,
        "gate_only_0p25": 0.6560045220988296,
        "gate_only_0p5": 0.6628703864448104,
        "gate_only_1": 0.6708695734737913,
        "gate_only_1p5": 0.6489378920554548
      }
    },
    "overall": {
      "alpha_vs_bt_art": {
        "p": 0.08156745017844687,
        "r": 0.09191894960517397
      },
      "alpha_vs_output_rv": {
        "p": 0.37353716902216855,
        "r": -0.04703774438227699
      },
      "bt_art_rate_by_condition": {
        "bridge_only_2": 0.125,
        "bridge_only_3": 0.2916666666666667,
        "control": 0.1527777777777778,
        "gate_0p25_bridge_2": 0.2222222222222222,
        "gate_0p25_bridge_3": 0.24999999999999997,
        "gate_0p5_bridge_2": 0.25,
        "gate_0p5_bridge_3": 0.26388888888888884,
        "gate_1_bridge_2": 0.2916666666666667,
        "gate_1_bridge_3": 0.25,
        "gate_1p5_bridge_2": 0.15277777777777776,
        "gate_1p5_bridge_3": 0.15277777777777776,
        "gate_only_0p25": 0.20833333333333334,
        "gate_only_0p5": 0.13888888888888887,
        "gate_only_1": 0.18055555555555555,
        "gate_only_1p5": 0.125
      },
      "mean_output_rv_by_condition": {
        "bridge_only_2": 0.6546882025081365,
        "bridge_only_3": 0.6433257203397431,
        "control": 0.6440649331000059,
        "gate_0p25_bridge_2": 0.6621617009238662,
        "gate_0p25_bridge_3": 0.6215792863271329,
        "gate_0p5_bridge_2": 0.6159014477076391,
        "gate_0p5_bridge_3": 0.6281944063476091,
        "gate_1_bridge_2": 0.6302863973140448,
        "gate_1_bridge_3": 0.6440417739479195,
        "gate_1p5_bridge_2": 0.6632875074190006,
        "gate_1p5_bridge_3": 0.6509436177524899,
        "gate_only_0p25": 0.6577203303811624,
        "gate_only_0p5": 0.6582486618176001,
        "gate_only_1": 0.648209685002214,
        "gate_only_1p5": 0.6594951725977144
      }
    },
    "recursive": {
      "alpha_vs_bt_art": {
        "p": 0.10732389611502433,
        "r": 0.12042931795819252
      },
      "alpha_vs_output_rv": {
        "p": 0.28097501475758335,
        "r": -0.08079059532067924
      },
      "bt_art_rate_by_condition": {
        "bridge_only_2": 0.16666666666666666,
        "bridge_only_3": 0.44444444444444436,
        "control": 0.25,
        "gate_0p25_bridge_2": 0.3055555555555555,
        "gate_0p25_bridge_3": 0.3888888888888889,
        "gate_0p5_bridge_2": 0.3611111111111111,
        "gate_0p5_bridge_3": 0.3888888888888889,
        "gate_1_bridge_2": 0.4166666666666667,
        "gate_1_bridge_3": 0.4166666666666666,
        "gate_1p5_bridge_2": 0.22222222222222224,
        "gate_1p5_bridge_3": 0.2222222222222222,
        "gate_only_0p25": 0.2777777777777778,
        "gate_only_0p5": 0.2222222222222222,
        "gate_only_1": 0.3055555555555556,
        "gate_only_1p5": 0.22222222222222224
      },
      "mean_output_rv_by_condition": {
        "bridge_only_2": 0.6713644001897968,
        "bridge_only_3": 0.6430804655506567,
        "control": 0.657076428282222,
        "gate_0p25_bridge_2": 0.6462943369992098,
        "gate_0p25_bridge_3": 0.63164902471185,
        "gate_0p5_bridge_2": 0.6251525512492606,
        "gate_0p5_bridge_3": 0.6198703005466769,
        "gate_1_bridge_2": 0.616735540070145,
        "gate_1_bridge_3": 0.6247833965541906,
        "gate_1p5_bridge_2": 0.6695065181140384,
        "gate_1p5_bridge_3": 0.6506684645639574,
        "gate_only_0p25": 0.6594361386634953,
        "gate_only_0p5": 0.6536269371903897,
        "gate_only_1": 0.6255497965306368,
        "gate_only_1p5": 0.670052453139974
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
      "gate_0p25_bridge_2": {
        "alpha": 2.25,
        "bt_art_cohens_h": 0.28790424673741494,
        "bt_art_exact_sign_p": 0.6875,
        "bt_art_prompt_losses": 2,
        "bt_art_prompt_wins": 4,
        "bt_art_rate_control": 0.05555555555555555,
        "bt_art_rate_delta": 0.08333333333333333,
        "bt_art_rate_delta_ci_95": [
          -0.08333333333333333,
          0.22222222222222224
        ],
        "bt_art_rate_treated": 0.13888888888888887,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": 0.44226038278607904,
        "rv_delta_ci_95": [
          -0.008378523553998828,
          0.10724212095089732
        ],
        "rv_delta_mean": 0.04697562693073259,
        "rv_p_value": 0.1537539992139281
      },
      "gate_0p25_bridge_3": {
        "alpha": 3.25,
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
        "rv_cohens_dz": -0.27739117030620947,
        "rv_delta_ci_95": [
          -0.05888988093829366,
          0.018020609159916934
        ],
        "rv_delta_mean": -0.01954388997537405,
        "rv_p_value": 0.3572434721018992
      },
      "gate_0p5_bridge_2": {
        "alpha": 2.5,
        "bt_art_cohens_h": 0.28790424673741494,
        "bt_art_exact_sign_p": 0.375,
        "bt_art_prompt_losses": 1,
        "bt_art_prompt_wins": 4,
        "bt_art_rate_control": 0.05555555555555555,
        "bt_art_rate_delta": 0.08333333333333333,
        "bt_art_rate_delta_ci_95": [
          -0.027777777777777776,
          0.19444444444444442
        ],
        "bt_art_rate_treated": 0.13888888888888887,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": -0.3927836765550224,
        "rv_delta_ci_95": [
          -0.05719414303620531,
          0.011554503965718629
        ],
        "rv_delta_mean": -0.024403093751772183,
        "rv_p_value": 0.20085355534444152
      },
      "gate_0p5_bridge_3": {
        "alpha": 3.5,
        "bt_art_cohens_h": 0.28790424673741494,
        "bt_art_exact_sign_p": 0.375,
        "bt_art_prompt_losses": 1,
        "bt_art_prompt_wins": 4,
        "bt_art_rate_control": 0.05555555555555555,
        "bt_art_rate_delta": 0.08333333333333333,
        "bt_art_rate_delta_ci_95": [
          -0.027777777777777776,
          0.19444444444444442
        ],
        "bt_art_rate_treated": 0.13888888888888887,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": 0.06658860956863888,
        "rv_delta_ci_95": [
          -0.037234468346405175,
          0.04987582334440726
        ],
        "rv_delta_mean": 0.0054650742307515016,
        "rv_p_value": 0.8218054054133219
      },
      "gate_1_bridge_2": {
        "alpha": 3.0,
        "bt_art_cohens_h": 0.3651864209075137,
        "bt_art_exact_sign_p": 0.453125,
        "bt_art_prompt_losses": 2,
        "bt_art_prompt_wins": 5,
        "bt_art_rate_control": 0.05555555555555555,
        "bt_art_rate_delta": 0.11111111111111112,
        "bt_art_rate_delta_ci_95": [
          -0.05555555555555555,
          0.27777777777777773
        ],
        "bt_art_rate_treated": 0.16666666666666666,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": 0.17829114489229247,
        "rv_delta_ci_95": [
          -0.023753738307259895,
          0.051367017615825995
        ],
        "rv_delta_mean": 0.012783816640154697,
        "rv_p_value": 0.5493989540397589
      },
      "gate_1_bridge_3": {
        "alpha": 4.0,
        "bt_art_cohens_h": 0.10980329379673442,
        "bt_art_exact_sign_p": 1.0,
        "bt_art_prompt_losses": 2,
        "bt_art_prompt_wins": 3,
        "bt_art_rate_control": 0.05555555555555555,
        "bt_art_rate_delta": 0.027777777777777776,
        "bt_art_rate_delta_ci_95": [
          -0.08333333333333333,
          0.13888888888888887
        ],
        "bt_art_rate_treated": 0.08333333333333333,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": 0.3856050102200872,
        "rv_delta_ci_95": [
          -0.01266576822820323,
          0.07639728051297576
        ],
        "rv_delta_mean": 0.03224671342385844,
        "rv_p_value": 0.20860227850283258
      },
      "gate_1p5_bridge_2": {
        "alpha": 3.5,
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
        "rv_cohens_dz": 0.258035116562652,
        "rv_delta_ci_95": [
          -0.029012661943609144,
          0.08058948124814155
        ],
        "rv_delta_mean": 0.026015058806172917,
        "rv_p_value": 0.3905347299338149
      },
      "gate_1p5_bridge_3": {
        "alpha": 4.5,
        "bt_art_cohens_h": 0.10980329379673442,
        "bt_art_exact_sign_p": 1.0,
        "bt_art_prompt_losses": 1,
        "bt_art_prompt_wins": 2,
        "bt_art_rate_control": 0.05555555555555555,
        "bt_art_rate_delta": 0.027777777777777776,
        "bt_art_rate_delta_ci_95": [
          -0.05555555555555555,
          0.13888888888888887
        ],
        "bt_art_rate_treated": 0.08333333333333333,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": 0.19388275993186557,
        "rv_delta_ci_95": [
          -0.030378438573394714,
          0.0782740336097192
        ],
        "rv_delta_mean": 0.020165333023232573,
        "rv_p_value": 0.5156768236219698
      },
      "gate_only_0p25": {
        "alpha": 0.25,
        "bt_art_cohens_h": 0.28790424673741494,
        "bt_art_exact_sign_p": 0.375,
        "bt_art_prompt_losses": 1,
        "bt_art_prompt_wins": 4,
        "bt_art_rate_control": 0.05555555555555555,
        "bt_art_rate_delta": 0.08333333333333333,
        "bt_art_rate_delta_ci_95": [
          -0.027777777777777776,
          0.19444444444444442
        ],
        "bt_art_rate_treated": 0.13888888888888887,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": 0.2431784521176963,
        "rv_delta_ci_95": [
          -0.026645822308374995,
          0.08214169133997117
        ],
        "rv_delta_mean": 0.02495108418103992,
        "rv_p_value": 0.41750658330456936
      },
      "gate_only_0p5": {
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
        "rv_cohens_dz": 0.4080983931321849,
        "rv_delta_ci_95": [
          -0.009707484522632875,
          0.07545539443407473
        ],
        "rv_delta_mean": 0.031816948527020696,
        "rv_p_value": 0.18512239243610998
      },
      "gate_only_1": {
        "alpha": 1.0,
        "bt_art_cohens_h": 0.0,
        "bt_art_exact_sign_p": 1.0,
        "bt_art_prompt_losses": 1,
        "bt_art_prompt_wins": 1,
        "bt_art_rate_control": 0.05555555555555555,
        "bt_art_rate_delta": 0.0,
        "bt_art_rate_delta_ci_95": [
          -0.08333333333333333,
          0.08333333333333333
        ],
        "bt_art_rate_treated": 0.05555555555555555,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": 0.3683363110929146,
        "rv_delta_ci_95": [
          -0.018863105836303348,
          0.09575340221974941
        ],
        "rv_delta_mean": 0.03981613555600141,
        "rv_p_value": 0.22825566221964055
      },
      "gate_only_1p5": {
        "alpha": 1.5,
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
        "rv_cohens_dz": 0.13579742902367054,
        "rv_delta_ci_95": [
          -0.051105084762310155,
          0.08857199751726158
        ],
        "rv_delta_mean": 0.017884454137665068,
        "rv_p_value": 0.6472472353273313
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
      "gate_0p25_bridge_2": {
        "alpha": 2.25,
        "bt_art_cohens_h": 0.178616551191757,
        "bt_art_exact_sign_p": 0.5810546875,
        "bt_art_prompt_losses": 5,
        "bt_art_prompt_wins": 8,
        "bt_art_rate_control": 0.1527777777777778,
        "bt_art_rate_delta": 0.06944444444444443,
        "bt_art_rate_delta_ci_95": [
          -0.041666666666666664,
          0.19444444444444442
        ],
        "bt_art_rate_treated": 0.2222222222222222,
        "n_prompt_pairs": 24,
        "rv_cohens_dz": 0.16048800107055688,
        "rv_delta_ci_95": [
          -0.02676541780149317,
          0.06332213573085872
        ],
        "rv_delta_mean": 0.0180967678238602,
        "rv_p_value": 0.4397600522410222
      },
      "gate_0p25_bridge_3": {
        "alpha": 3.25,
        "bt_art_cohens_h": 0.2440487458097319,
        "bt_art_exact_sign_p": 0.266845703125,
        "bt_art_prompt_losses": 4,
        "bt_art_prompt_wins": 9,
        "bt_art_rate_control": 0.1527777777777778,
        "bt_art_rate_delta": 0.09722222222222221,
        "bt_art_rate_delta_ci_95": [
          -0.041666666666666664,
          0.23611111111111108
        ],
        "bt_art_rate_treated": 0.24999999999999997,
        "n_prompt_pairs": 24,
        "rv_cohens_dz": -0.21258360911283636,
        "rv_delta_ci_95": [
          -0.064825636918853,
          0.018071066630142785
        ],
        "rv_delta_mean": -0.022485646772872986,
        "rv_p_value": 0.30849427479877284
      },
      "gate_0p5_bridge_2": {
        "alpha": 2.5,
        "bt_art_cohens_h": 0.24404874580973213,
        "bt_art_exact_sign_p": 0.14599609375,
        "bt_art_prompt_losses": 3,
        "bt_art_prompt_wins": 9,
        "bt_art_rate_control": 0.1527777777777778,
        "bt_art_rate_delta": 0.09722222222222221,
        "bt_art_rate_delta_ci_95": [
          -4.625929269271485e-18,
          0.19444444444444442
        ],
        "bt_art_rate_treated": 0.25,
        "n_prompt_pairs": 24,
        "rv_cohens_dz": -0.3061541834941622,
        "rv_delta_ci_95": [
          -0.06487949737049933,
          0.0075578908300209776
        ],
        "rv_delta_mean": -0.028163485392366783,
        "rv_p_value": 0.14725778329477157
      },
      "gate_0p5_bridge_3": {
        "alpha": 3.5,
        "bt_art_cohens_h": 0.27583742684258616,
        "bt_art_exact_sign_p": 0.1795654296875,
        "bt_art_prompt_losses": 4,
        "bt_art_prompt_wins": 10,
        "bt_art_rate_control": 0.1527777777777778,
        "bt_art_rate_delta": 0.11111111111111112,
        "bt_art_rate_delta_ci_95": [
          0.0,
          0.23611111111111108
        ],
        "bt_art_rate_treated": 0.26388888888888884,
        "n_prompt_pairs": 24,
        "rv_cohens_dz": -0.18295213976307614,
        "rv_delta_ci_95": [
          -0.04958672577062014,
          0.016476611412775018
        ],
        "rv_delta_mean": -0.015870526752396738,
        "rv_p_value": 0.3793893685260714
      },
      "gate_1_bridge_2": {
        "alpha": 3.0,
        "bt_art_cohens_h": 0.3378720901035034,
        "bt_art_exact_sign_p": 0.09228515625,
        "bt_art_prompt_losses": 3,
        "bt_art_prompt_wins": 10,
        "bt_art_rate_control": 0.1527777777777778,
        "bt_art_rate_delta": 0.13888888888888887,
        "bt_art_rate_delta_ci_95": [
          0.027777777777777773,
          0.2503472222222235
        ],
        "bt_art_rate_treated": 0.2916666666666667,
        "n_prompt_pairs": 24,
        "rv_cohens_dz": -0.13305471839979927,
        "rv_delta_ci_95": [
          -0.05251771464209884,
          0.02739937281630072
        ],
        "rv_delta_mean": -0.013778535785961124,
        "rv_p_value": 0.5209704297352761
      },
      "gate_1_bridge_3": {
        "alpha": 4.0,
        "bt_art_cohens_h": 0.24404874580973213,
        "bt_art_exact_sign_p": 0.4239501953125,
        "bt_art_prompt_losses": 5,
        "bt_art_prompt_wins": 9,
        "bt_art_rate_control": 0.1527777777777778,
        "bt_art_rate_delta": 0.09722222222222222,
        "bt_art_rate_delta_ci_95": [
          -0.041666666666666664,
          0.23611111111111113
        ],
        "bt_art_rate_treated": 0.25,
        "n_prompt_pairs": 24,
        "rv_cohens_dz": -0.00022705400852041836,
        "rv_delta_ci_95": [
          -0.04024814893723634,
          0.03759467101166413
        ],
        "rv_delta_mean": -2.3159152086451678e-05,
        "rv_p_value": 0.999122078562106
      },
      "gate_1p5_bridge_2": {
        "alpha": 3.5,
        "bt_art_cohens_h": -1.1102230246251565e-16,
        "bt_art_exact_sign_p": 1.0,
        "bt_art_prompt_losses": 6,
        "bt_art_prompt_wins": 6,
        "bt_art_rate_control": 0.1527777777777778,
        "bt_art_rate_delta": 0.0,
        "bt_art_rate_delta_ci_95": [
          -0.11111111111111112,
          0.12499999999999999
        ],
        "bt_art_rate_treated": 0.15277777777777776,
        "n_prompt_pairs": 24,
        "rv_cohens_dz": 0.16905914871512887,
        "rv_delta_ci_95": [
          -0.02442908648115175,
          0.06459479581996573
        ],
        "rv_delta_mean": 0.019222574318994625,
        "rv_p_value": 0.41605928853989815
      },
      "gate_1p5_bridge_3": {
        "alpha": 4.5,
        "bt_art_cohens_h": -1.1102230246251565e-16,
        "bt_art_exact_sign_p": 1.0,
        "bt_art_prompt_losses": 5,
        "bt_art_prompt_wins": 5,
        "bt_art_rate_control": 0.1527777777777778,
        "bt_art_rate_delta": 2.3129646346357427e-18,
        "bt_art_rate_delta_ci_95": [
          -0.1111111111111111,
          0.1111111111111111
        ],
        "bt_art_rate_treated": 0.15277777777777776,
        "n_prompt_pairs": 24,
        "rv_cohens_dz": 0.06225377665945128,
        "rv_delta_ci_95": [
          -0.034951839045369804,
          0.05009642503157967
        ],
        "rv_delta_mean": 0.006878684652483968,
        "rv_p_value": 0.7631241767447585
      },
      "gate_only_0p25": {
        "alpha": 0.25,
        "bt_art_cohens_h": 0.14482093599602797,
        "bt_art_exact_sign_p": 0.3876953125,
        "bt_art_prompt_losses": 4,
        "bt_art_prompt_wins": 8,
        "bt_art_rate_control": 0.1527777777777778,
        "bt_art_rate_delta": 0.05555555555555556,
        "bt_art_rate_delta_ci_95": [
          -0.041666666666666664,
          0.15277777777777776
        ],
        "bt_art_rate_treated": 0.20833333333333334,
        "n_prompt_pairs": 24,
        "rv_cohens_dz": 0.16238075628080684,
        "rv_delta_ci_95": [
          -0.017769139158447734,
          0.0481477227643748
        ],
        "rv_delta_mean": 0.013655397281156577,
        "rv_p_value": 0.43445621616641605
      },
      "gate_only_0p5": {
        "alpha": 0.5,
        "bt_art_cohens_h": -0.039362308989034256,
        "bt_art_exact_sign_p": 1.0,
        "bt_art_prompt_losses": 6,
        "bt_art_prompt_wins": 6,
        "bt_art_rate_control": 0.1527777777777778,
        "bt_art_rate_delta": -0.0138888888888889,
        "bt_art_rate_delta_ci_95": [
          -0.125,
          0.08333333333333333
        ],
        "bt_art_rate_treated": 0.13888888888888887,
        "n_prompt_pairs": 24,
        "rv_cohens_dz": 0.1369930178965023,
        "rv_delta_ci_95": [
          -0.025699542560358318,
          0.05495775310783216
        ],
        "rv_delta_mean": 0.014183728717594178,
        "rv_p_value": 0.5088248508977756
      },
      "gate_only_1": {
        "alpha": 1.0,
        "bt_art_cohens_h": 0.07459443982805836,
        "bt_art_exact_sign_p": 0.5078125,
        "bt_art_prompt_losses": 3,
        "bt_art_prompt_wins": 6,
        "bt_art_rate_control": 0.1527777777777778,
        "bt_art_rate_delta": 0.027777777777777776,
        "bt_art_rate_delta_ci_95": [
          -0.06944444444444446,
          0.1111111111111111
        ],
        "bt_art_rate_treated": 0.18055555555555555,
        "n_prompt_pairs": 24,
        "rv_cohens_dz": 0.03838561109406521,
        "rv_delta_ci_95": [
          -0.03659577065209357,
          0.045683020795672415
        ],
        "rv_delta_mean": 0.004144751902208134,
        "rv_p_value": 0.8524864633851864
      },
      "gate_only_1p5": {
        "alpha": 1.5,
        "bt_art_cohens_h": -0.08041455757345006,
        "bt_art_exact_sign_p": 1.0,
        "bt_art_prompt_losses": 5,
        "bt_art_prompt_wins": 5,
        "bt_art_rate_control": 0.1527777777777778,
        "bt_art_rate_delta": -0.027777777777777773,
        "bt_art_rate_delta_ci_95": [
          -0.1388888888888889,
          0.06944444444444443
        ],
        "bt_art_rate_treated": 0.125,
        "n_prompt_pairs": 24,
        "rv_cohens_dz": 0.12435714041505114,
        "rv_delta_ci_95": [
          -0.03339058984780262,
          0.06286587521375715
        ],
        "rv_delta_mean": 0.015430239497708482,
        "rv_p_value": 0.5483478255146983
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
      "gate_0p25_bridge_2": {
        "alpha": 2.25,
        "bt_art_cohens_h": 0.12417353571770406,
        "bt_art_exact_sign_p": 1.0,
        "bt_art_prompt_losses": 3,
        "bt_art_prompt_wins": 4,
        "bt_art_rate_control": 0.25,
        "bt_art_rate_delta": 0.055555555555555546,
        "bt_art_rate_delta_ci_95": [
          -0.1111111111111111,
          0.2222222222222222
        ],
        "bt_art_rate_treated": 0.3055555555555555,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": -0.0928516380422219,
        "rv_delta_ci_95": [
          -0.07313420808562238,
          0.05250714194959944
        ],
        "rv_delta_mean": -0.010782091283012185,
        "rv_p_value": 0.7537490884692777
      },
      "gate_0p25_bridge_3": {
        "alpha": 3.25,
        "bt_art_cohens_h": 0.29950568329692784,
        "bt_art_exact_sign_p": 0.5078125,
        "bt_art_prompt_losses": 3,
        "bt_art_prompt_wins": 6,
        "bt_art_rate_control": 0.25,
        "bt_art_rate_delta": 0.13888888888888887,
        "bt_art_rate_delta_ci_95": [
          -0.1111111111111111,
          0.38888888888888884
        ],
        "bt_art_rate_treated": 0.3888888888888889,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": -0.18740195173386498,
        "rv_delta_ci_95": [
          -0.09932268414641321,
          0.04649128429052392
        ],
        "rv_delta_mean": -0.025427403570371928,
        "rv_p_value": 0.5295437178870993
      },
      "gate_0p5_bridge_2": {
        "alpha": 2.5,
        "bt_art_cohens_h": 0.2421187023674538,
        "bt_art_exact_sign_p": 0.453125,
        "bt_art_prompt_losses": 2,
        "bt_art_prompt_wins": 5,
        "bt_art_rate_control": 0.25,
        "bt_art_rate_delta": 0.11111111111111109,
        "bt_art_rate_delta_ci_95": [
          -0.05555555555555556,
          0.2777777777777778
        ],
        "bt_art_rate_treated": 0.3611111111111111,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": -0.27172210039016204,
        "rv_delta_ci_95": [
          -0.09553558955341976,
          0.02900528979462495
        ],
        "rv_delta_mean": -0.031923877032961394,
        "rv_p_value": 0.36677710811960906
      },
      "gate_0p5_bridge_3": {
        "alpha": 3.5,
        "bt_art_cohens_h": 0.29950568329692784,
        "bt_art_exact_sign_p": 0.5078125,
        "bt_art_prompt_losses": 3,
        "bt_art_prompt_wins": 6,
        "bt_art_rate_control": 0.25,
        "bt_art_rate_delta": 0.13888888888888887,
        "bt_art_rate_delta_ci_95": [
          -0.05555555555555555,
          0.3340277777777804
        ],
        "bt_art_rate_treated": 0.3888888888888889,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": -0.41584447057339086,
        "rv_delta_ci_95": [
          -0.08942926689884988,
          0.006482988734429857
        ],
        "rv_delta_mean": -0.03720612773554498,
        "rv_p_value": 0.17756919461195347
      },
      "gate_1_bridge_2": {
        "alpha": 3.0,
        "bt_art_cohens_h": 0.3561506963786094,
        "bt_art_exact_sign_p": 0.21875,
        "bt_art_prompt_losses": 1,
        "bt_art_prompt_wins": 5,
        "bt_art_rate_control": 0.25,
        "bt_art_rate_delta": 0.16666666666666666,
        "bt_art_rate_delta_ci_95": [
          -4.625929269271485e-18,
          0.3333333333333334
        ],
        "bt_art_rate_treated": 0.4166666666666667,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": -0.3215279678445799,
        "rv_delta_ci_95": [
          -0.10955438472479005,
          0.027231959495243225
        ],
        "rv_delta_mean": -0.04034088821207694,
        "rv_p_value": 0.2891082855164956
      },
      "gate_1_bridge_3": {
        "alpha": 4.0,
        "bt_art_cohens_h": 0.3561506963786092,
        "bt_art_exact_sign_p": 0.5078125,
        "bt_art_prompt_losses": 3,
        "bt_art_prompt_wins": 6,
        "bt_art_rate_control": 0.25,
        "bt_art_rate_delta": 0.16666666666666666,
        "bt_art_rate_delta_ci_95": [
          -0.1111111111111111,
          0.4166666666666667
        ],
        "bt_art_rate_treated": 0.4166666666666666,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": -0.2889790817021223,
        "rv_delta_ci_95": [
          -0.09685315576968728,
          0.023697125938661204
        ],
        "rv_delta_mean": -0.03229303172803134,
        "rv_p_value": 0.3383137154547836
      },
      "gate_1p5_bridge_2": {
        "alpha": 3.5,
        "bt_art_cohens_h": -0.06543219461797511,
        "bt_art_exact_sign_p": 1.0,
        "bt_art_prompt_losses": 5,
        "bt_art_prompt_wins": 4,
        "bt_art_rate_control": 0.25,
        "bt_art_rate_delta": -0.027777777777777776,
        "bt_art_rate_delta_ci_95": [
          -0.22222222222222224,
          0.19444444444444442
        ],
        "bt_art_rate_treated": 0.22222222222222224,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": 0.09599493916540082,
        "rv_delta_ci_95": [
          -0.059964824572469114,
          0.08153653586401904
        ],
        "rv_delta_mean": 0.012430089831816338,
        "rv_p_value": 0.7457353375244458
      },
      "gate_1p5_bridge_3": {
        "alpha": 4.5,
        "bt_art_cohens_h": -0.06543219461797511,
        "bt_art_exact_sign_p": 1.0,
        "bt_art_prompt_losses": 4,
        "bt_art_prompt_wins": 3,
        "bt_art_rate_control": 0.25,
        "bt_art_rate_delta": -0.027777777777777776,
        "bt_art_rate_delta_ci_95": [
          -0.22222222222222224,
          0.16666666666666666
        ],
        "bt_art_rate_treated": 0.2222222222222222,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": -0.0535391863898187,
        "rv_delta_ci_95": [
          -0.06780528540779861,
          0.05576971345129081
        ],
        "rv_delta_mean": -0.006407963718264635,
        "rv_p_value": 0.8562397580532104
      },
      "gate_only_0p25": {
        "alpha": 0.25,
        "bt_art_cohens_h": 0.06304478391697654,
        "bt_art_exact_sign_p": 1.0,
        "bt_art_prompt_losses": 3,
        "bt_art_prompt_wins": 4,
        "bt_art_rate_control": 0.25,
        "bt_art_rate_delta": 0.027777777777777776,
        "bt_art_rate_delta_ci_95": [
          -0.1111111111111111,
          0.16666666666666666
        ],
        "bt_art_rate_treated": 0.2777777777777778,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": 0.03739992609209776,
        "rv_delta_ci_95": [
          -0.03360158427939889,
          0.03414383443966663
        ],
        "rv_delta_mean": 0.0023597103812732314,
        "rv_p_value": 0.899255489187341
      },
      "gate_only_0p5": {
        "alpha": 0.5,
        "bt_art_cohens_h": -0.06543219461797511,
        "bt_art_exact_sign_p": 1.0,
        "bt_art_prompt_losses": 4,
        "bt_art_prompt_wins": 4,
        "bt_art_rate_control": 0.25,
        "bt_art_rate_delta": -0.027777777777777766,
        "bt_art_rate_delta_ci_95": [
          -0.2222222222222222,
          0.13888888888888887
        ],
        "bt_art_rate_treated": 0.2222222222222222,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": -0.027567519007107782,
        "rv_delta_ci_95": [
          -0.07069615798364752,
          0.06433592701833647
        ],
        "rv_delta_mean": -0.0034494910918323415,
        "rv_p_value": 0.9256378309479443
      },
      "gate_only_1": {
        "alpha": 1.0,
        "bt_art_cohens_h": 0.12417353571770429,
        "bt_art_exact_sign_p": 0.453125,
        "bt_art_prompt_losses": 2,
        "bt_art_prompt_wins": 5,
        "bt_art_rate_control": 0.25,
        "bt_art_rate_delta": 0.055555555555555546,
        "bt_art_rate_delta_ci_95": [
          -0.1388888888888889,
          0.19444444444444445
        ],
        "bt_art_rate_treated": 0.3055555555555556,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": -0.3165895108850447,
        "rv_delta_ci_95": [
          -0.08932174722075159,
          0.02024626764700226
        ],
        "rv_delta_mean": -0.031526631751585134,
        "rv_p_value": 0.2962017561823178
      },
      "gate_only_1p5": {
        "alpha": 1.5,
        "bt_art_cohens_h": -0.06543219461797511,
        "bt_art_exact_sign_p": 1.0,
        "bt_art_prompt_losses": 4,
        "bt_art_prompt_wins": 5,
        "bt_art_rate_control": 0.25,
        "bt_art_rate_delta": -0.027777777777777804,
        "bt_art_rate_delta_ci_95": [
          -0.25,
          0.16666666666666666
        ],
        "bt_art_rate_treated": 0.22222222222222224,
        "n_prompt_pairs": 12,
        "rv_cohens_dz": 0.10654198155402848,
        "rv_delta_ci_95": [
          -0.05686638181118582,
          0.0750060486300591
        ],
        "rv_delta_mean": 0.012976024857751895,
        "rv_p_value": 0.7190795159279978
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
        "gate": 0.0
      },
      "name": "control"
    },
    {
      "alphas": {
        "bridge": 2.0,
        "gate": 0.0
      },
      "name": "bridge_only_2"
    },
    {
      "alphas": {
        "bridge": 3.0,
        "gate": 0.0
      },
      "name": "bridge_only_3"
    },
    {
      "alphas": {
        "bridge": 0.0,
        "gate": 0.25
      },
      "name": "gate_only_0p25"
    },
    {
      "alphas": {
        "bridge": 0.0,
        "gate": 0.5
      },
      "name": "gate_only_0p5"
    },
    {
      "alphas": {
        "bridge": 0.0,
        "gate": 1.0
      },
      "name": "gate_only_1"
    },
    {
      "alphas": {
        "bridge": 0.0,
        "gate": 1.5
      },
      "name": "gate_only_1p5"
    },
    {
      "alphas": {
        "bridge": 2.0,
        "gate": 0.25
      },
      "name": "gate_0p25_bridge_2"
    },
    {
      "alphas": {
        "bridge": 2.0,
        "gate": 0.5
      },
      "name": "gate_0p5_bridge_2"
    },
    {
      "alphas": {
        "bridge": 2.0,
        "gate": 1.0
      },
      "name": "gate_1_bridge_2"
    },
    {
      "alphas": {
        "bridge": 2.0,
        "gate": 1.5
      },
      "name": "gate_1p5_bridge_2"
    },
    {
      "alphas": {
        "bridge": 3.0,
        "gate": 0.25
      },
      "name": "gate_0p25_bridge_3"
    },
    {
      "alphas": {
        "bridge": 3.0,
        "gate": 0.5
      },
      "name": "gate_0p5_bridge_3"
    },
    {
      "alphas": {
        "bridge": 3.0,
        "gate": 1.0
      },
      "name": "gate_1_bridge_3"
    },
    {
      "alphas": {
        "bridge": 3.0,
        "gate": 1.5
      },
      "name": "gate_1p5_bridge_3"
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
    "gate": {
      "centroid_cosine": 0.9719824194908142,
      "component": "residual",
      "direction_norm": 0.9240488409996033,
      "layer": 5,
      "token_window": 16,
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
  "timestamp": "20260312_091557",
  "top_p": 0.95,
  "verdict": "multisite_sufficient"
}
```
