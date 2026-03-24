# Run report: causal_state_targeted_scan_v1

- **run_dir**: `results/phase1_mechanism/runs/20260315_062302_causal_state_targeted_scan_v1_qwen_base_targeted_scan_v2_deterministic`
- **prompt_bank_version**: `2ac959a313614329`

## Summary (machine-readable)

```json
{
  "artifacts": {
    "best_candidate_json": "results/phase1_mechanism/runs/20260315_062302_causal_state_targeted_scan_v1_qwen_base_targeted_scan_v2_deterministic/best_candidate.json",
    "candidate_records_jsonl": "results/phase1_mechanism/runs/20260315_062302_causal_state_targeted_scan_v1_qwen_base_targeted_scan_v2_deterministic/candidate_records.jsonl",
    "candidate_scores_json": "results/phase1_mechanism/runs/20260315_062302_causal_state_targeted_scan_v1_qwen_base_targeted_scan_v2_deterministic/candidate_scores.json",
    "config": "results/phase1_mechanism/runs/20260315_062302_causal_state_targeted_scan_v1_qwen_base_targeted_scan_v2_deterministic/config.json",
    "manifest": "results/phase1_mechanism/runs/20260315_062302_causal_state_targeted_scan_v1_qwen_base_targeted_scan_v2_deterministic/manifest.json",
    "report": "results/phase1_mechanism/runs/20260315_062302_causal_state_targeted_scan_v1_qwen_base_targeted_scan_v2_deterministic/report.md",
    "shared_control_records_jsonl": "results/phase1_mechanism/runs/20260315_062302_causal_state_targeted_scan_v1_qwen_base_targeted_scan_v2_deterministic/shared_control_records.jsonl",
    "summary": "results/phase1_mechanism/runs/20260315_062302_causal_state_targeted_scan_v1_qwen_base_targeted_scan_v2_deterministic/summary.json"
  },
  "best_candidate": {
    "alpha": 2.0,
    "by_condition": {
      "away": {
        "alpha": -2.0,
        "by_prompt_mode": {
          "baseline": {
            "bt_art_rate": 0.0,
            "class_counts": {
              "REPETITIVE": 1,
              "SURFACE": 11
            },
            "mean_generated_tokens": 81.08333333333333,
            "mean_output_rv": 1.1838547181144867,
            "n": 12,
            "std_output_rv": 0.22890244618932976
          },
          "recursive": {
            "bt_art_rate": 0.0,
            "class_counts": {
              "SURFACE": 12
            },
            "mean_generated_tokens": 60.833333333333336,
            "mean_output_rv": 1.1730643198720432,
            "n": 12,
            "std_output_rv": 0.2004627958288119
          }
        },
        "overall": {
          "bt_art_rate": 0.0,
          "class_counts": {
            "REPETITIVE": 1,
            "SURFACE": 23
          },
          "mean_generated_tokens": 70.95833333333333,
          "mean_output_rv": 1.1794116129558336,
          "n": 24,
          "std_output_rv": 0.2111217647255228
        }
      },
      "none": {
        "alpha": 0.0,
        "by_prompt_mode": {
          "baseline": {
            "bt_art_rate": 0.0,
            "class_counts": {
              "REPETITIVE": 1,
              "SURFACE": 11
            },
            "mean_generated_tokens": 81.08333333333333,
            "mean_output_rv": 1.1816740999209578,
            "n": 12,
            "std_output_rv": 0.22856714554613797
          },
          "recursive": {
            "bt_art_rate": 0.0,
            "class_counts": {
              "SURFACE": 12
            },
            "mean_generated_tokens": 61.333333333333336,
            "mean_output_rv": 1.206636453889282,
            "n": 12,
            "std_output_rv": 0.19718131597462538
          }
        },
        "overall": {
          "bt_art_rate": 0.0,
          "class_counts": {
            "REPETITIVE": 1,
            "SURFACE": 23
          },
          "mean_generated_tokens": 71.20833333333333,
          "mean_output_rv": 1.1919527162608559,
          "n": 24,
          "std_output_rv": 0.21006473303167564
        }
      },
      "toward": {
        "alpha": 2.0,
        "by_prompt_mode": {
          "baseline": {
            "bt_art_rate": 0.0,
            "class_counts": {
              "REPETITIVE": 1,
              "SURFACE": 11
            },
            "mean_generated_tokens": 73.66666666666667,
            "mean_output_rv": 1.1794873040764238,
            "n": 12,
            "std_output_rv": 0.24217240237750925
          },
          "recursive": {
            "bt_art_rate": 0.0,
            "class_counts": {
              "SURFACE": 12
            },
            "mean_generated_tokens": 54.333333333333336,
            "mean_output_rv": 1.2205194886893511,
            "n": 12,
            "std_output_rv": 0.1761748417085741
          }
        },
        "overall": {
          "bt_art_rate": 0.0,
          "class_counts": {
            "REPETITIVE": 1,
            "SURFACE": 23
          },
          "mean_generated_tokens": 64.0,
          "mean_output_rv": 1.1959001779215948,
          "n": 24,
          "std_output_rv": 0.21220420735501
        }
      }
    },
    "candidate_name": "L21_W32_A2",
    "effects_by_prompt_mode": {
      "baseline": {
        "away": {
          "alpha": -2.0,
          "bt_art_cohens_h": 0.0,
          "bt_art_exact_sign_p": null,
          "bt_art_prompt_losses": 0,
          "bt_art_prompt_wins": 0,
          "bt_art_rate_control": 0.0,
          "bt_art_rate_delta": 0.0,
          "bt_art_rate_delta_ci_95": [
            0.0,
            0.0
          ],
          "bt_art_rate_treated": 0.0,
          "n_prompt_pairs": 12,
          "rv_cohens_dz": 0.31622776601683794,
          "rv_delta_ci_95": [
            0.0,
            0.0065418545805867815
          ],
          "rv_delta_mean": 0.002180618193528927,
          "rv_p_value": 0.3434363961379136
        },
        "toward": {
          "alpha": 2.0,
          "bt_art_cohens_h": 0.0,
          "bt_art_exact_sign_p": null,
          "bt_art_prompt_losses": 0,
          "bt_art_prompt_wins": 0,
          "bt_art_rate_control": 0.0,
          "bt_art_rate_delta": 0.0,
          "bt_art_rate_delta_ci_95": [
            0.0,
            0.0
          ],
          "bt_art_rate_treated": 0.0,
          "n_prompt_pairs": 12,
          "rv_cohens_dz": 0.36110440054868387,
          "rv_delta_ci_95": [
            0.0,
            0.006963362253777299
          ],
          "rv_delta_mean": 0.0024359952743467053,
          "rv_p_value": 0.3102364932122943
        }
      },
      "overall": {
        "away": {
          "alpha": -2.0,
          "bt_art_cohens_h": 0.0,
          "bt_art_exact_sign_p": null,
          "bt_art_prompt_losses": 0,
          "bt_art_prompt_wins": 0,
          "bt_art_rate_control": 0.0,
          "bt_art_rate_delta": 0.0,
          "bt_art_rate_delta_ci_95": [
            0.0,
            0.0
          ],
          "bt_art_rate_treated": 0.0,
          "n_prompt_pairs": 24,
          "rv_cohens_dz": -0.1330917663495874,
          "rv_delta_ci_95": [
            -0.06170204457185378,
            0.02155471883079403
          ],
          "rv_delta_mean": -0.012541103305022613,
          "rv_p_value": 0.5907521607494315
        },
        "toward": {
          "alpha": 2.0,
          "bt_art_cohens_h": 0.0,
          "bt_art_exact_sign_p": null,
          "bt_art_prompt_losses": 0,
          "bt_art_prompt_wins": 0,
          "bt_art_rate_control": 0.0,
          "bt_art_rate_delta": 0.0,
          "bt_art_rate_delta_ci_95": [
            0.0,
            0.0
          ],
          "bt_art_rate_treated": 0.0,
          "n_prompt_pairs": 24,
          "rv_cohens_dz": 0.19516495234819667,
          "rv_delta_ci_95": [
            -0.021766702274683393,
            0.06765212118538547
          ],
          "rv_delta_mean": 0.01853954861321908,
          "rv_p_value": 0.4622562163570921
        }
      },
      "recursive": {
        "away": {
          "alpha": -2.0,
          "bt_art_cohens_h": 0.0,
          "bt_art_exact_sign_p": null,
          "bt_art_prompt_losses": 0,
          "bt_art_prompt_wins": 0,
          "bt_art_rate_control": 0.0,
          "bt_art_rate_delta": 0.0,
          "bt_art_rate_delta_ci_95": [
            0.0,
            0.0
          ],
          "bt_art_rate_treated": 0.0,
          "n_prompt_pairs": 12,
          "rv_cohens_dz": -0.22268365760571787,
          "rv_delta_ci_95": [
            -0.14992781386406673,
            0.05188824478025341
          ],
          "rv_delta_mean": -0.033572134017239096,
          "rv_p_value": 0.5772437276283461
        },
        "toward": {
          "alpha": 2.0,
          "bt_art_cohens_h": 0.0,
          "bt_art_exact_sign_p": null,
          "bt_art_prompt_losses": 0,
          "bt_art_prompt_wins": 0,
          "bt_art_rate_control": 0.0,
          "bt_art_rate_delta": 0.0,
          "bt_art_rate_delta_ci_95": [
            0.0,
            0.0
          ],
          "bt_art_rate_treated": 0.0,
          "n_prompt_pairs": 12,
          "rv_cohens_dz": 0.2754383946596949,
          "rv_delta_ci_95": [
            -0.058140188837716066,
            0.16886144798571864
          ],
          "rv_delta_mean": 0.042694878621527634,
          "rv_p_value": 0.5298088310056192
        }
      }
    },
    "objective": {
      "score": -0.00024359952743467053,
      "score_breakdown": {
        "baseline_rv_penalty": 0.0024359952743467053,
        "baseline_spillover_penalty": 0.0,
        "recursive_bt_gain": 0.0,
        "recursive_bt_suppression": -0.0,
        "recursive_rv_alignment": 0.0
      },
      "sign_checks": {
        "recursive_away_bt_negative": false,
        "recursive_away_rv_positive": false,
        "recursive_toward_bt_positive": false,
        "recursive_toward_rv_negative": false
      },
      "sign_match_count": 0
    },
    "rank": 1,
    "source_layer": 21,
    "state_source": {
      "centroid_cosine": 0.9066073894500732,
      "negative_centroid_norm": 80.34516143798828,
      "negative_selected_n": 52,
      "positive_centroid_norm": 90.68419647216797,
      "positive_selected_n": 72,
      "raw_direction_norm": 38.3121452331543
    },
    "window": 32
  },
  "bootstrap_resamples": 1000,
  "candidate_rankings": [
    {
      "alpha": 2.0,
      "by_condition": {
        "away": {
          "alpha": -2.0,
          "by_prompt_mode": {
            "baseline": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "REPETITIVE": 1,
                "SURFACE": 11
              },
              "mean_generated_tokens": 81.08333333333333,
              "mean_output_rv": 1.1838547181144867,
              "n": 12,
              "std_output_rv": 0.22890244618932976
            },
            "recursive": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "SURFACE": 12
              },
              "mean_generated_tokens": 60.833333333333336,
              "mean_output_rv": 1.1730643198720432,
              "n": 12,
              "std_output_rv": 0.2004627958288119
            }
          },
          "overall": {
            "bt_art_rate": 0.0,
            "class_counts": {
              "REPETITIVE": 1,
              "SURFACE": 23
            },
            "mean_generated_tokens": 70.95833333333333,
            "mean_output_rv": 1.1794116129558336,
            "n": 24,
            "std_output_rv": 0.2111217647255228
          }
        },
        "none": {
          "alpha": 0.0,
          "by_prompt_mode": {
            "baseline": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "REPETITIVE": 1,
                "SURFACE": 11
              },
              "mean_generated_tokens": 81.08333333333333,
              "mean_output_rv": 1.1816740999209578,
              "n": 12,
              "std_output_rv": 0.22856714554613797
            },
            "recursive": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "SURFACE": 12
              },
              "mean_generated_tokens": 61.333333333333336,
              "mean_output_rv": 1.206636453889282,
              "n": 12,
              "std_output_rv": 0.19718131597462538
            }
          },
          "overall": {
            "bt_art_rate": 0.0,
            "class_counts": {
              "REPETITIVE": 1,
              "SURFACE": 23
            },
            "mean_generated_tokens": 71.20833333333333,
            "mean_output_rv": 1.1919527162608559,
            "n": 24,
            "std_output_rv": 0.21006473303167564
          }
        },
        "toward": {
          "alpha": 2.0,
          "by_prompt_mode": {
            "baseline": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "REPETITIVE": 1,
                "SURFACE": 11
              },
              "mean_generated_tokens": 73.66666666666667,
              "mean_output_rv": 1.1794873040764238,
              "n": 12,
              "std_output_rv": 0.24217240237750925
            },
            "recursive": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "SURFACE": 12
              },
              "mean_generated_tokens": 54.333333333333336,
              "mean_output_rv": 1.2205194886893511,
              "n": 12,
              "std_output_rv": 0.1761748417085741
            }
          },
          "overall": {
            "bt_art_rate": 0.0,
            "class_counts": {
              "REPETITIVE": 1,
              "SURFACE": 23
            },
            "mean_generated_tokens": 64.0,
            "mean_output_rv": 1.1959001779215948,
            "n": 24,
            "std_output_rv": 0.21220420735501
          }
        }
      },
      "candidate_name": "L21_W32_A2",
      "effects_by_prompt_mode": {
        "baseline": {
          "away": {
            "alpha": -2.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": 0.31622776601683794,
            "rv_delta_ci_95": [
              0.0,
              0.0065418545805867815
            ],
            "rv_delta_mean": 0.002180618193528927,
            "rv_p_value": 0.3434363961379136
          },
          "toward": {
            "alpha": 2.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": 0.36110440054868387,
            "rv_delta_ci_95": [
              0.0,
              0.006963362253777299
            ],
            "rv_delta_mean": 0.0024359952743467053,
            "rv_p_value": 0.3102364932122943
          }
        },
        "overall": {
          "away": {
            "alpha": -2.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 24,
            "rv_cohens_dz": -0.1330917663495874,
            "rv_delta_ci_95": [
              -0.06170204457185378,
              0.02155471883079403
            ],
            "rv_delta_mean": -0.012541103305022613,
            "rv_p_value": 0.5907521607494315
          },
          "toward": {
            "alpha": 2.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 24,
            "rv_cohens_dz": 0.19516495234819667,
            "rv_delta_ci_95": [
              -0.021766702274683393,
              0.06765212118538547
            ],
            "rv_delta_mean": 0.01853954861321908,
            "rv_p_value": 0.4622562163570921
          }
        },
        "recursive": {
          "away": {
            "alpha": -2.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": -0.22268365760571787,
            "rv_delta_ci_95": [
              -0.14992781386406673,
              0.05188824478025341
            ],
            "rv_delta_mean": -0.033572134017239096,
            "rv_p_value": 0.5772437276283461
          },
          "toward": {
            "alpha": 2.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": 0.2754383946596949,
            "rv_delta_ci_95": [
              -0.058140188837716066,
              0.16886144798571864
            ],
            "rv_delta_mean": 0.042694878621527634,
            "rv_p_value": 0.5298088310056192
          }
        }
      },
      "objective": {
        "score": -0.00024359952743467053,
        "score_breakdown": {
          "baseline_rv_penalty": 0.0024359952743467053,
          "baseline_spillover_penalty": 0.0,
          "recursive_bt_gain": 0.0,
          "recursive_bt_suppression": -0.0,
          "recursive_rv_alignment": 0.0
        },
        "sign_checks": {
          "recursive_away_bt_negative": false,
          "recursive_away_rv_positive": false,
          "recursive_toward_bt_positive": false,
          "recursive_toward_rv_negative": false
        },
        "sign_match_count": 0
      },
      "rank": 1,
      "source_layer": 21,
      "state_source": {
        "centroid_cosine": 0.9066073894500732,
        "negative_centroid_norm": 80.34516143798828,
        "negative_selected_n": 52,
        "positive_centroid_norm": 90.68419647216797,
        "positive_selected_n": 72,
        "raw_direction_norm": 38.3121452331543
      },
      "window": 32
    },
    {
      "alpha": 3.0,
      "by_condition": {
        "away": {
          "alpha": -3.0,
          "by_prompt_mode": {
            "baseline": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "REPETITIVE": 1,
                "SURFACE": 11
              },
              "mean_generated_tokens": 73.66666666666667,
              "mean_output_rv": 1.1935958440382695,
              "n": 12,
              "std_output_rv": 0.2317858917413827
            },
            "recursive": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "SURFACE": 12
              },
              "mean_generated_tokens": 60.916666666666664,
              "mean_output_rv": 1.1427968583991717,
              "n": 12,
              "std_output_rv": 0.16759773874509046
            }
          },
          "overall": {
            "bt_art_rate": 0.0,
            "class_counts": {
              "REPETITIVE": 1,
              "SURFACE": 23
            },
            "mean_generated_tokens": 67.29166666666667,
            "mean_output_rv": 1.171371287821164,
            "n": 24,
            "std_output_rv": 0.20141043886991256
          }
        },
        "none": {
          "alpha": 0.0,
          "by_prompt_mode": {
            "baseline": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "REPETITIVE": 1,
                "SURFACE": 11
              },
              "mean_generated_tokens": 81.08333333333333,
              "mean_output_rv": 1.1816740999209578,
              "n": 12,
              "std_output_rv": 0.22856714554613797
            },
            "recursive": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "SURFACE": 12
              },
              "mean_generated_tokens": 61.333333333333336,
              "mean_output_rv": 1.206636453889282,
              "n": 12,
              "std_output_rv": 0.19718131597462538
            }
          },
          "overall": {
            "bt_art_rate": 0.0,
            "class_counts": {
              "REPETITIVE": 1,
              "SURFACE": 23
            },
            "mean_generated_tokens": 71.20833333333333,
            "mean_output_rv": 1.1919527162608559,
            "n": 24,
            "std_output_rv": 0.21006473303167564
          }
        },
        "toward": {
          "alpha": 3.0,
          "by_prompt_mode": {
            "baseline": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "REPETITIVE": 1,
                "SURFACE": 11
              },
              "mean_generated_tokens": 81.08333333333333,
              "mean_output_rv": 1.1778314877412244,
              "n": 12,
              "std_output_rv": 0.19350129154461326
            },
            "recursive": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "SURFACE": 12
              },
              "mean_generated_tokens": 54.333333333333336,
              "mean_output_rv": 1.2090577943761094,
              "n": 12,
              "std_output_rv": 0.17721723260426925
            }
          },
          "overall": {
            "bt_art_rate": 0.0,
            "class_counts": {
              "REPETITIVE": 1,
              "SURFACE": 23
            },
            "mean_generated_tokens": 67.70833333333333,
            "mean_output_rv": 1.1895413527293064,
            "n": 24,
            "std_output_rv": 0.18214848237489994
          }
        }
      },
      "candidate_name": "L23_W32_A3",
      "effects_by_prompt_mode": {
        "baseline": {
          "away": {
            "alpha": -3.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": 0.5153026085756349,
            "rv_delta_ci_95": [
              0.002362336376323008,
              0.03796583657971294
            ],
            "rv_delta_mean": 0.01654453523619205,
            "rv_p_value": 0.16071051258458285
          },
          "toward": {
            "alpha": 3.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": -0.037745515759993784,
            "rv_delta_ci_95": [
              -0.07092085754738056,
              0.055159433105370526
            ],
            "rv_delta_mean": -0.00384261217973334,
            "rv_p_value": 0.9076106151586852
          }
        },
        "overall": {
          "away": {
            "alpha": -3.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 24,
            "rv_cohens_dz": -0.18120514257989057,
            "rv_delta_ci_95": [
              -0.07725216420342128,
              0.020366850375499512
            ],
            "rv_delta_mean": -0.01862352195656527,
            "rv_p_value": 0.4797169526137992
          },
          "toward": {
            "alpha": 3.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 24,
            "rv_cohens_dz": 0.07576318877440649,
            "rv_delta_ci_95": [
              -0.04968719389508521,
              0.07088332421948525
            ],
            "rv_delta_mean": 0.00931081150327389,
            "rv_p_value": 0.766012071664668
          }
        },
        "recursive": {
          "away": {
            "alpha": -3.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": -0.44272934640858375,
            "rv_delta_ci_95": [
              -0.16659951702085202,
              0.016893948992062917
            ],
            "rv_delta_mean": -0.0638395954901104,
            "rv_p_value": 0.28586392979932285
          },
          "toward": {
            "alpha": 3.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": 0.19471061994526023,
            "rv_delta_ci_95": [
              -0.08315343965341926,
              0.1483024418866095
            ],
            "rv_delta_mean": 0.03123318430828594,
            "rv_p_value": 0.6535200449491552
          }
        }
      },
      "objective": {
        "score": -0.00038426121797333404,
        "score_breakdown": {
          "baseline_rv_penalty": 0.00384261217973334,
          "baseline_spillover_penalty": 0.0,
          "recursive_bt_gain": 0.0,
          "recursive_bt_suppression": -0.0,
          "recursive_rv_alignment": 0.0
        },
        "sign_checks": {
          "recursive_away_bt_negative": false,
          "recursive_away_rv_positive": false,
          "recursive_toward_bt_positive": false,
          "recursive_toward_rv_negative": false
        },
        "sign_match_count": 0
      },
      "rank": 2,
      "source_layer": 23,
      "state_source": {
        "centroid_cosine": 0.8981044888496399,
        "negative_centroid_norm": 119.8793716430664,
        "negative_selected_n": 52,
        "positive_centroid_norm": 130.3688507080078,
        "positive_selected_n": 72,
        "raw_direction_norm": 57.402034759521484
      },
      "window": 32
    },
    {
      "alpha": 2.0,
      "by_condition": {
        "away": {
          "alpha": -2.0,
          "by_prompt_mode": {
            "baseline": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "REPETITIVE": 1,
                "SURFACE": 11
              },
              "mean_generated_tokens": 81.08333333333333,
              "mean_output_rv": 1.1872616726821943,
              "n": 12,
              "std_output_rv": 0.23037307427619538
            },
            "recursive": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "SURFACE": 12
              },
              "mean_generated_tokens": 61.25,
              "mean_output_rv": 1.151546076339546,
              "n": 12,
              "std_output_rv": 0.1744551567911669
            }
          },
          "overall": {
            "bt_art_rate": 0.0,
            "class_counts": {
              "REPETITIVE": 1,
              "SURFACE": 23
            },
            "mean_generated_tokens": 71.16666666666667,
            "mean_output_rv": 1.172555250658751,
            "n": 24,
            "std_output_rv": 0.20394635889059654
          }
        },
        "none": {
          "alpha": 0.0,
          "by_prompt_mode": {
            "baseline": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "REPETITIVE": 1,
                "SURFACE": 11
              },
              "mean_generated_tokens": 81.08333333333333,
              "mean_output_rv": 1.1816740999209578,
              "n": 12,
              "std_output_rv": 0.22856714554613797
            },
            "recursive": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "SURFACE": 12
              },
              "mean_generated_tokens": 61.333333333333336,
              "mean_output_rv": 1.206636453889282,
              "n": 12,
              "std_output_rv": 0.19718131597462538
            }
          },
          "overall": {
            "bt_art_rate": 0.0,
            "class_counts": {
              "REPETITIVE": 1,
              "SURFACE": 23
            },
            "mean_generated_tokens": 71.20833333333333,
            "mean_output_rv": 1.1919527162608559,
            "n": 24,
            "std_output_rv": 0.21006473303167564
          }
        },
        "toward": {
          "alpha": 2.0,
          "by_prompt_mode": {
            "baseline": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "REPETITIVE": 1,
                "SURFACE": 11
              },
              "mean_generated_tokens": 81.08333333333333,
              "mean_output_rv": 1.187173917710552,
              "n": 12,
              "std_output_rv": 0.22962231010095557
            },
            "recursive": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "SURFACE": 12
              },
              "mean_generated_tokens": 61.333333333333336,
              "mean_output_rv": 1.2444885684012659,
              "n": 12,
              "std_output_rv": 0.18198899488009473
            }
          },
          "overall": {
            "bt_art_rate": 0.0,
            "class_counts": {
              "REPETITIVE": 1,
              "SURFACE": 23
            },
            "mean_generated_tokens": 71.20833333333333,
            "mean_output_rv": 1.210774067994964,
            "n": 24,
            "std_output_rv": 0.20718107371748687
          }
        }
      },
      "candidate_name": "L25_W8_A2",
      "effects_by_prompt_mode": {
        "baseline": {
          "away": {
            "alpha": -2.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": 0.31622776601683794,
            "rv_delta_ci_95": [
              0.0,
              0.01676271828370961
            ],
            "rv_delta_mean": 0.005587572761236536,
            "rv_p_value": 0.3434363961379136
          },
          "toward": {
            "alpha": 2.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": 0.3263192541343123,
            "rv_delta_ci_95": [
              0.0,
              0.01618929215644609
            ],
            "rv_delta_mean": 0.005499817789594208,
            "rv_p_value": 0.3290451246368585
          }
        },
        "overall": {
          "away": {
            "alpha": -2.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 24,
            "rv_cohens_dz": -0.21053528624092321,
            "rv_delta_ci_95": [
              -0.06910326693564281,
              0.011004033726265466
            ],
            "rv_delta_mean": -0.01939746560210518,
            "rv_p_value": 0.39819528855011094
          },
          "toward": {
            "alpha": 2.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 24,
            "rv_cohens_dz": 0.21055948495329396,
            "rv_delta_ci_95": [
              -0.018580620332943373,
              0.0603924683496768
            ],
            "rv_delta_mean": 0.01882135173410755,
            "rv_p_value": 0.3981422888989946
          }
        },
        "recursive": {
          "away": {
            "alpha": -2.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": -0.39308744596183354,
            "rv_delta_ci_95": [
              -0.1635746621992248,
              0.015549449546490322
            ],
            "rv_delta_mean": -0.05509037754973621,
            "rv_p_value": 0.3384169003087605
          },
          "toward": {
            "alpha": 2.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": 0.2665739161941498,
            "rv_delta_ci_95": [
              -0.05724605020594405,
              0.1329876709549761
            ],
            "rv_delta_mean": 0.037852114511983746,
            "rv_p_value": 0.5070740399556327
          }
        }
      },
      "objective": {
        "score": -0.0005499817789594209,
        "score_breakdown": {
          "baseline_rv_penalty": 0.005499817789594208,
          "baseline_spillover_penalty": 0.0,
          "recursive_bt_gain": 0.0,
          "recursive_bt_suppression": -0.0,
          "recursive_rv_alignment": 0.0
        },
        "sign_checks": {
          "recursive_away_bt_negative": false,
          "recursive_away_rv_positive": false,
          "recursive_toward_bt_positive": false,
          "recursive_toward_rv_negative": false
        },
        "sign_match_count": 0
      },
      "rank": 3,
      "source_layer": 25,
      "state_source": {
        "centroid_cosine": 0.9202046990394592,
        "negative_centroid_norm": 216.7076873779297,
        "negative_selected_n": 57,
        "positive_centroid_norm": 224.44369506835938,
        "positive_selected_n": 72,
        "raw_direction_norm": 88.44281005859375
      },
      "window": 8
    },
    {
      "alpha": 2.0,
      "by_condition": {
        "away": {
          "alpha": -2.0,
          "by_prompt_mode": {
            "baseline": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "REPETITIVE": 1,
                "SURFACE": 11
              },
              "mean_generated_tokens": 73.66666666666667,
              "mean_output_rv": 1.177051308802077,
              "n": 12,
              "std_output_rv": 0.24193572039190303
            },
            "recursive": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "SURFACE": 12
              },
              "mean_generated_tokens": 61.333333333333336,
              "mean_output_rv": 1.1545749442689908,
              "n": 12,
              "std_output_rv": 0.17747207771404513
            }
          },
          "overall": {
            "bt_art_rate": 0.0,
            "class_counts": {
              "REPETITIVE": 1,
              "SURFACE": 23
            },
            "mean_generated_tokens": 67.5,
            "mean_output_rv": 1.1672178993188518,
            "n": 24,
            "std_output_rv": 0.20963942795294427
          }
        },
        "none": {
          "alpha": 0.0,
          "by_prompt_mode": {
            "baseline": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "REPETITIVE": 1,
                "SURFACE": 11
              },
              "mean_generated_tokens": 81.08333333333333,
              "mean_output_rv": 1.1816740999209578,
              "n": 12,
              "std_output_rv": 0.22856714554613797
            },
            "recursive": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "SURFACE": 12
              },
              "mean_generated_tokens": 61.333333333333336,
              "mean_output_rv": 1.206636453889282,
              "n": 12,
              "std_output_rv": 0.19718131597462538
            }
          },
          "overall": {
            "bt_art_rate": 0.0,
            "class_counts": {
              "REPETITIVE": 1,
              "SURFACE": 23
            },
            "mean_generated_tokens": 71.20833333333333,
            "mean_output_rv": 1.1919527162608559,
            "n": 24,
            "std_output_rv": 0.21006473303167564
          }
        },
        "toward": {
          "alpha": 2.0,
          "by_prompt_mode": {
            "baseline": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "REPETITIVE": 1,
                "SURFACE": 11
              },
              "mean_generated_tokens": 81.08333333333333,
              "mean_output_rv": 1.187173917710552,
              "n": 12,
              "std_output_rv": 0.22962231010095557
            },
            "recursive": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "SURFACE": 12
              },
              "mean_generated_tokens": 61.333333333333336,
              "mean_output_rv": 1.2119793749610577,
              "n": 12,
              "std_output_rv": 0.145950537934394
            }
          },
          "overall": {
            "bt_art_rate": 0.0,
            "class_counts": {
              "REPETITIVE": 1,
              "SURFACE": 23
            },
            "mean_generated_tokens": 71.20833333333333,
            "mean_output_rv": 1.197387929519584,
            "n": 24,
            "std_output_rv": 0.1944351815415904
          }
        }
      },
      "candidate_name": "L25_W16_A2",
      "effects_by_prompt_mode": {
        "baseline": {
          "away": {
            "alpha": -2.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": 0.0,
            "rv_delta_ci_95": [
              0.0,
              0.0
            ],
            "rv_delta_mean": 0.0,
            "rv_p_value": null
          },
          "toward": {
            "alpha": 2.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": 0.3263192541343123,
            "rv_delta_ci_95": [
              0.0,
              0.016344372762614356
            ],
            "rv_delta_mean": 0.005499817789594208,
            "rv_p_value": 0.3290451246368585
          }
        },
        "overall": {
          "away": {
            "alpha": -2.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 24,
            "rv_cohens_dz": -0.24692587067907132,
            "rv_delta_ci_95": [
              -0.07541630370528922,
              0.0070932105875410765
            ],
            "rv_delta_mean": -0.022776910458877468,
            "rv_p_value": 0.33896339248541013
          },
          "toward": {
            "alpha": 2.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 24,
            "rv_cohens_dz": 0.06577224011550238,
            "rv_delta_ci_95": [
              -0.026656069781045507,
              0.047639573657737896
            ],
            "rv_delta_mean": 0.005435213258727717,
            "rv_p_value": 0.789715660932086
          }
        },
        "recursive": {
          "away": {
            "alpha": -2.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": -0.3728898468699707,
            "rv_delta_ci_95": [
              -0.15823443819442282,
              0.016893948992062917
            ],
            "rv_delta_mean": -0.05206150962029136,
            "rv_p_value": 0.361949863823407
          },
          "toward": {
            "alpha": 2.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": 0.04006470262475891,
            "rv_delta_ci_95": [
              -0.06868425028973066,
              0.10358373050370225
            ],
            "rv_delta_mean": 0.005342921071775586,
            "rv_p_value": 0.919036534051135
          }
        }
      },
      "objective": {
        "score": -0.0005499817789594209,
        "score_breakdown": {
          "baseline_rv_penalty": 0.005499817789594208,
          "baseline_spillover_penalty": 0.0,
          "recursive_bt_gain": 0.0,
          "recursive_bt_suppression": -0.0,
          "recursive_rv_alignment": 0.0
        },
        "sign_checks": {
          "recursive_away_bt_negative": false,
          "recursive_away_rv_positive": false,
          "recursive_toward_bt_positive": false,
          "recursive_toward_rv_negative": false
        },
        "sign_match_count": 0
      },
      "rank": 4,
      "source_layer": 25,
      "state_source": {
        "centroid_cosine": 0.9261921644210815,
        "negative_centroid_norm": 218.0831756591797,
        "negative_selected_n": 57,
        "positive_centroid_norm": 224.69021606445312,
        "positive_selected_n": 72,
        "raw_direction_norm": 85.30532836914062
      },
      "window": 16
    },
    {
      "alpha": 2.0,
      "by_condition": {
        "away": {
          "alpha": -2.0,
          "by_prompt_mode": {
            "baseline": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "REPETITIVE": 1,
                "SURFACE": 11
              },
              "mean_generated_tokens": 81.08333333333333,
              "mean_output_rv": 1.1872616726821943,
              "n": 12,
              "std_output_rv": 0.23037307427619538
            },
            "recursive": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "SURFACE": 12
              },
              "mean_generated_tokens": 60.916666666666664,
              "mean_output_rv": 1.1545749442689908,
              "n": 12,
              "std_output_rv": 0.17747207771404513
            }
          },
          "overall": {
            "bt_art_rate": 0.0,
            "class_counts": {
              "REPETITIVE": 1,
              "SURFACE": 23
            },
            "mean_generated_tokens": 71.0,
            "mean_output_rv": 1.1738024315708753,
            "n": 24,
            "std_output_rv": 0.20479001378860073
          }
        },
        "none": {
          "alpha": 0.0,
          "by_prompt_mode": {
            "baseline": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "REPETITIVE": 1,
                "SURFACE": 11
              },
              "mean_generated_tokens": 81.08333333333333,
              "mean_output_rv": 1.1816740999209578,
              "n": 12,
              "std_output_rv": 0.22856714554613797
            },
            "recursive": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "SURFACE": 12
              },
              "mean_generated_tokens": 61.333333333333336,
              "mean_output_rv": 1.206636453889282,
              "n": 12,
              "std_output_rv": 0.19718131597462538
            }
          },
          "overall": {
            "bt_art_rate": 0.0,
            "class_counts": {
              "REPETITIVE": 1,
              "SURFACE": 23
            },
            "mean_generated_tokens": 71.20833333333333,
            "mean_output_rv": 1.1919527162608559,
            "n": 24,
            "std_output_rv": 0.21006473303167564
          }
        },
        "toward": {
          "alpha": 2.0,
          "by_prompt_mode": {
            "baseline": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "REPETITIVE": 1,
                "SURFACE": 11
              },
              "mean_generated_tokens": 81.08333333333333,
              "mean_output_rv": 1.187173917710552,
              "n": 12,
              "std_output_rv": 0.22962231010095557
            },
            "recursive": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "SURFACE": 12
              },
              "mean_generated_tokens": 61.333333333333336,
              "mean_output_rv": 1.2119793749610577,
              "n": 12,
              "std_output_rv": 0.145950537934394
            }
          },
          "overall": {
            "bt_art_rate": 0.0,
            "class_counts": {
              "REPETITIVE": 1,
              "SURFACE": 23
            },
            "mean_generated_tokens": 71.20833333333333,
            "mean_output_rv": 1.197387929519584,
            "n": 24,
            "std_output_rv": 0.1944351815415904
          }
        }
      },
      "candidate_name": "L25_W32_A2",
      "effects_by_prompt_mode": {
        "baseline": {
          "away": {
            "alpha": -2.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": 0.31622776601683794,
            "rv_delta_ci_95": [
              0.0,
              0.01676271828370961
            ],
            "rv_delta_mean": 0.005587572761236536,
            "rv_p_value": 0.3434363961379136
          },
          "toward": {
            "alpha": 2.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": 0.3263192541343123,
            "rv_delta_ci_95": [
              0.0,
              0.016344372762614356
            ],
            "rv_delta_mean": 0.005499817789594208,
            "rv_p_value": 0.3290451246368585
          }
        },
        "overall": {
          "away": {
            "alpha": -2.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 24,
            "rv_cohens_dz": -0.1987388434080987,
            "rv_delta_ci_95": [
              -0.06667268208800661,
              0.012422712113597342
            ],
            "rv_delta_mean": -0.018150284689980832,
            "rv_p_value": 0.42458451614664916
          },
          "toward": {
            "alpha": 2.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 24,
            "rv_cohens_dz": 0.06577224011550238,
            "rv_delta_ci_95": [
              -0.028569174062603613,
              0.04548114002954133
            ],
            "rv_delta_mean": 0.005435213258727717,
            "rv_p_value": 0.789715660932086
          }
        },
        "recursive": {
          "away": {
            "alpha": -2.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": -0.3728898468699707,
            "rv_delta_ci_95": [
              -0.16323479316092512,
              0.016893948992062917
            ],
            "rv_delta_mean": -0.05206150962029136,
            "rv_p_value": 0.361949863823407
          },
          "toward": {
            "alpha": 2.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": 0.04006470262475891,
            "rv_delta_ci_95": [
              -0.06993659860060801,
              0.10342287326959201
            ],
            "rv_delta_mean": 0.005342921071775586,
            "rv_p_value": 0.919036534051135
          }
        }
      },
      "objective": {
        "score": -0.0005499817789594209,
        "score_breakdown": {
          "baseline_rv_penalty": 0.005499817789594208,
          "baseline_spillover_penalty": 0.0,
          "recursive_bt_gain": 0.0,
          "recursive_bt_suppression": -0.0,
          "recursive_rv_alignment": 0.0
        },
        "sign_checks": {
          "recursive_away_bt_negative": false,
          "recursive_away_rv_positive": false,
          "recursive_toward_bt_positive": false,
          "recursive_toward_rv_negative": false
        },
        "sign_match_count": 0
      },
      "rank": 5,
      "source_layer": 25,
      "state_source": {
        "centroid_cosine": 0.9391899108886719,
        "negative_centroid_norm": 216.47683715820312,
        "negative_selected_n": 52,
        "positive_centroid_norm": 226.54112243652344,
        "positive_selected_n": 72,
        "raw_direction_norm": 77.88246154785156
      },
      "window": 32
    },
    {
      "alpha": 3.0,
      "by_condition": {
        "away": {
          "alpha": -3.0,
          "by_prompt_mode": {
            "baseline": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "REPETITIVE": 1,
                "SURFACE": 11
              },
              "mean_generated_tokens": 73.66666666666667,
              "mean_output_rv": 1.177051308802077,
              "n": 12,
              "std_output_rv": 0.24193572039190303
            },
            "recursive": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "SURFACE": 12
              },
              "mean_generated_tokens": 61.25,
              "mean_output_rv": 1.1592318126076222,
              "n": 12,
              "std_output_rv": 0.1842847187126521
            }
          },
          "overall": {
            "bt_art_rate": 0.0,
            "class_counts": {
              "REPETITIVE": 1,
              "SURFACE": 23
            },
            "mean_generated_tokens": 67.45833333333333,
            "mean_output_rv": 1.169255279217003,
            "n": 24,
            "std_output_rv": 0.21186136655749016
          }
        },
        "none": {
          "alpha": 0.0,
          "by_prompt_mode": {
            "baseline": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "REPETITIVE": 1,
                "SURFACE": 11
              },
              "mean_generated_tokens": 81.08333333333333,
              "mean_output_rv": 1.1816740999209578,
              "n": 12,
              "std_output_rv": 0.22856714554613797
            },
            "recursive": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "SURFACE": 12
              },
              "mean_generated_tokens": 61.333333333333336,
              "mean_output_rv": 1.206636453889282,
              "n": 12,
              "std_output_rv": 0.19718131597462538
            }
          },
          "overall": {
            "bt_art_rate": 0.0,
            "class_counts": {
              "REPETITIVE": 1,
              "SURFACE": 23
            },
            "mean_generated_tokens": 71.20833333333333,
            "mean_output_rv": 1.1919527162608559,
            "n": 24,
            "std_output_rv": 0.21006473303167564
          }
        },
        "toward": {
          "alpha": 3.0,
          "by_prompt_mode": {
            "baseline": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "REPETITIVE": 1,
                "SURFACE": 11
              },
              "mean_generated_tokens": 81.08333333333333,
              "mean_output_rv": 1.187173917710552,
              "n": 12,
              "std_output_rv": 0.22962231010095557
            },
            "recursive": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "SURFACE": 12
              },
              "mean_generated_tokens": 61.333333333333336,
              "mean_output_rv": 1.2407170098160825,
              "n": 12,
              "std_output_rv": 0.18148350289817528
            }
          },
          "overall": {
            "bt_art_rate": 0.0,
            "class_counts": {
              "REPETITIVE": 1,
              "SURFACE": 23
            },
            "mean_generated_tokens": 71.20833333333333,
            "mean_output_rv": 1.2092210732834179,
            "n": 24,
            "std_output_rv": 0.20675467761109353
          }
        }
      },
      "candidate_name": "L25_W8_A3",
      "effects_by_prompt_mode": {
        "baseline": {
          "away": {
            "alpha": -3.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": 0.0,
            "rv_delta_ci_95": [
              0.0,
              0.0
            ],
            "rv_delta_mean": 0.0,
            "rv_p_value": null
          },
          "toward": {
            "alpha": 3.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": 0.3263192541343123,
            "rv_delta_ci_95": [
              0.0,
              0.01618929215644609
            ],
            "rv_delta_mean": 0.005499817789594208,
            "rv_p_value": 0.3290451246368585
          }
        },
        "overall": {
          "away": {
            "alpha": -3.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 24,
            "rv_cohens_dz": -0.21852835716417343,
            "rv_delta_ci_95": [
              -0.07274035172703687,
              0.011265912490276472
            ],
            "rv_delta_mean": -0.020739530560726298,
            "rv_p_value": 0.3958361835823102
          },
          "toward": {
            "alpha": 3.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 24,
            "rv_cohens_dz": 0.18686474107664433,
            "rv_delta_ci_95": [
              -0.022517930877539644,
              0.06287485787161207
            ],
            "rv_delta_mean": 0.017268357022561437,
            "rv_p_value": 0.4522525433119059
          }
        },
        "recursive": {
          "away": {
            "alpha": -3.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": -0.32678961473650814,
            "rv_delta_ci_95": [
              -0.15781668659433853,
              0.03092092208264252
            ],
            "rv_delta_mean": -0.04740464128166011,
            "rv_p_value": 0.4204739790963055
          },
          "toward": {
            "alpha": 3.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": 0.2308962773441104,
            "rv_delta_ci_95": [
              -0.06276546203249114,
              0.1305587960657137
            ],
            "rv_delta_mean": 0.034080555926800336,
            "rv_p_value": 0.5636801993138877
          }
        }
      },
      "objective": {
        "score": -0.0005499817789594209,
        "score_breakdown": {
          "baseline_rv_penalty": 0.005499817789594208,
          "baseline_spillover_penalty": 0.0,
          "recursive_bt_gain": 0.0,
          "recursive_bt_suppression": -0.0,
          "recursive_rv_alignment": 0.0
        },
        "sign_checks": {
          "recursive_away_bt_negative": false,
          "recursive_away_rv_positive": false,
          "recursive_toward_bt_positive": false,
          "recursive_toward_rv_negative": false
        },
        "sign_match_count": 0
      },
      "rank": 6,
      "source_layer": 25,
      "state_source": {
        "centroid_cosine": 0.9202046990394592,
        "negative_centroid_norm": 216.7076873779297,
        "negative_selected_n": 57,
        "positive_centroid_norm": 224.44369506835938,
        "positive_selected_n": 72,
        "raw_direction_norm": 88.44281005859375
      },
      "window": 8
    },
    {
      "alpha": 3.0,
      "by_condition": {
        "away": {
          "alpha": -3.0,
          "by_prompt_mode": {
            "baseline": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "REPETITIVE": 1,
                "SURFACE": 11
              },
              "mean_generated_tokens": 73.66666666666667,
              "mean_output_rv": 1.1794742179059983,
              "n": 12,
              "std_output_rv": 0.24234272695302822
            },
            "recursive": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "SURFACE": 12
              },
              "mean_generated_tokens": 61.25,
              "mean_output_rv": 1.151546076339546,
              "n": 12,
              "std_output_rv": 0.1744551567911669
            }
          },
          "overall": {
            "bt_art_rate": 0.0,
            "class_counts": {
              "REPETITIVE": 1,
              "SURFACE": 23
            },
            "mean_generated_tokens": 67.45833333333333,
            "mean_output_rv": 1.1672556559706755,
            "n": 24,
            "std_output_rv": 0.2090484416572129
          }
        },
        "none": {
          "alpha": 0.0,
          "by_prompt_mode": {
            "baseline": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "REPETITIVE": 1,
                "SURFACE": 11
              },
              "mean_generated_tokens": 81.08333333333333,
              "mean_output_rv": 1.1816740999209578,
              "n": 12,
              "std_output_rv": 0.22856714554613797
            },
            "recursive": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "SURFACE": 12
              },
              "mean_generated_tokens": 61.333333333333336,
              "mean_output_rv": 1.206636453889282,
              "n": 12,
              "std_output_rv": 0.19718131597462538
            }
          },
          "overall": {
            "bt_art_rate": 0.0,
            "class_counts": {
              "REPETITIVE": 1,
              "SURFACE": 23
            },
            "mean_generated_tokens": 71.20833333333333,
            "mean_output_rv": 1.1919527162608559,
            "n": 24,
            "std_output_rv": 0.21006473303167564
          }
        },
        "toward": {
          "alpha": 3.0,
          "by_prompt_mode": {
            "baseline": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "REPETITIVE": 1,
                "SURFACE": 11
              },
              "mean_generated_tokens": 81.08333333333333,
              "mean_output_rv": 1.187173917710552,
              "n": 12,
              "std_output_rv": 0.22962231010095557
            },
            "recursive": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "SURFACE": 12
              },
              "mean_generated_tokens": 61.0,
              "mean_output_rv": 1.2400810500558666,
              "n": 12,
              "std_output_rv": 0.17761280059066703
            }
          },
          "overall": {
            "bt_art_rate": 0.0,
            "class_counts": {
              "REPETITIVE": 1,
              "SURFACE": 23
            },
            "mean_generated_tokens": 71.04166666666667,
            "mean_output_rv": 1.2089592074997995,
            "n": 24,
            "std_output_rv": 0.20544790418854703
          }
        }
      },
      "candidate_name": "L25_W16_A3",
      "effects_by_prompt_mode": {
        "baseline": {
          "away": {
            "alpha": -3.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": 0.3333333333333333,
            "rv_delta_ci_95": [
              0.0,
              0.00726872731176309
            ],
            "rv_delta_mean": 0.00242290910392103,
            "rv_p_value": 0.34659350708733416
          },
          "toward": {
            "alpha": 3.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": 0.3263192541343123,
            "rv_delta_ci_95": [
              0.0,
              0.016344372762614356
            ],
            "rv_delta_mean": 0.005499817789594208,
            "rv_p_value": 0.3290451246368585
          }
        },
        "overall": {
          "away": {
            "alpha": -3.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 24,
            "rv_cohens_dz": -0.24304952114672648,
            "rv_delta_ci_95": [
              -0.07241990692314612,
              0.008374768395579095
            ],
            "rv_delta_mean": -0.022739153807054012,
            "rv_p_value": 0.34636952258657694
          },
          "toward": {
            "alpha": 3.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 24,
            "rv_cohens_dz": 0.19163259394543486,
            "rv_delta_ci_95": [
              -0.019776170091850467,
              0.06155017048826544
            ],
            "rv_delta_mean": 0.017006491238943124,
            "rv_p_value": 0.441010740111464
          }
        },
        "recursive": {
          "away": {
            "alpha": -3.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": -0.39308744596183354,
            "rv_delta_ci_95": [
              -0.16626366109037,
              0.016893948992062917
            ],
            "rv_delta_mean": -0.05509037754973621,
            "rv_p_value": 0.3384169003087605
          },
          "toward": {
            "alpha": 3.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": 0.23625328568174075,
            "rv_delta_ci_95": [
              -0.057941678324539364,
              0.14465791449934023
            ],
            "rv_delta_mean": 0.03344459616658443,
            "rv_p_value": 0.5549384271291751
          }
        }
      },
      "objective": {
        "score": -0.0005499817789594209,
        "score_breakdown": {
          "baseline_rv_penalty": 0.005499817789594208,
          "baseline_spillover_penalty": 0.0,
          "recursive_bt_gain": 0.0,
          "recursive_bt_suppression": -0.0,
          "recursive_rv_alignment": 0.0
        },
        "sign_checks": {
          "recursive_away_bt_negative": false,
          "recursive_away_rv_positive": false,
          "recursive_toward_bt_positive": false,
          "recursive_toward_rv_negative": false
        },
        "sign_match_count": 0
      },
      "rank": 7,
      "source_layer": 25,
      "state_source": {
        "centroid_cosine": 0.9261921644210815,
        "negative_centroid_norm": 218.0831756591797,
        "negative_selected_n": 57,
        "positive_centroid_norm": 224.69021606445312,
        "positive_selected_n": 72,
        "raw_direction_norm": 85.30532836914062
      },
      "window": 16
    },
    {
      "alpha": 3.0,
      "by_condition": {
        "away": {
          "alpha": -3.0,
          "by_prompt_mode": {
            "baseline": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "REPETITIVE": 1,
                "SURFACE": 11
              },
              "mean_generated_tokens": 81.08333333333333,
              "mean_output_rv": 1.1937501021698464,
              "n": 12,
              "std_output_rv": 0.2341318753293023
            },
            "recursive": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "SURFACE": 12
              },
              "mean_generated_tokens": 60.916666666666664,
              "mean_output_rv": 1.151546076339546,
              "n": 12,
              "std_output_rv": 0.1744551567911669
            }
          },
          "overall": {
            "bt_art_rate": 0.0,
            "class_counts": {
              "REPETITIVE": 1,
              "SURFACE": 23
            },
            "mean_generated_tokens": 71.0,
            "mean_output_rv": 1.1763719738867817,
            "n": 24,
            "std_output_rv": 0.20665513173257619
          }
        },
        "none": {
          "alpha": 0.0,
          "by_prompt_mode": {
            "baseline": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "REPETITIVE": 1,
                "SURFACE": 11
              },
              "mean_generated_tokens": 81.08333333333333,
              "mean_output_rv": 1.1816740999209578,
              "n": 12,
              "std_output_rv": 0.22856714554613797
            },
            "recursive": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "SURFACE": 12
              },
              "mean_generated_tokens": 61.333333333333336,
              "mean_output_rv": 1.206636453889282,
              "n": 12,
              "std_output_rv": 0.19718131597462538
            }
          },
          "overall": {
            "bt_art_rate": 0.0,
            "class_counts": {
              "REPETITIVE": 1,
              "SURFACE": 23
            },
            "mean_generated_tokens": 71.20833333333333,
            "mean_output_rv": 1.1919527162608559,
            "n": 24,
            "std_output_rv": 0.21006473303167564
          }
        },
        "toward": {
          "alpha": 3.0,
          "by_prompt_mode": {
            "baseline": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "REPETITIVE": 1,
                "SURFACE": 11
              },
              "mean_generated_tokens": 73.66666666666667,
              "mean_output_rv": 1.1831622174571819,
              "n": 12,
              "std_output_rv": 0.24317925774051805
            },
            "recursive": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "SURFACE": 12
              },
              "mean_generated_tokens": 54.333333333333336,
              "mean_output_rv": 1.2163075668399743,
              "n": 12,
              "std_output_rv": 0.18362231128447534
            }
          },
          "overall": {
            "bt_art_rate": 0.0,
            "class_counts": {
              "REPETITIVE": 1,
              "SURFACE": 23
            },
            "mean_generated_tokens": 64.0,
            "mean_output_rv": 1.1964203572102992,
            "n": 24,
            "std_output_rv": 0.21474735791450464
          }
        }
      },
      "candidate_name": "L25_W32_A3",
      "effects_by_prompt_mode": {
        "baseline": {
          "away": {
            "alpha": -3.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": 0.31622776601683794,
            "rv_delta_ci_95": [
              0.0,
              0.03622800674666562
            ],
            "rv_delta_mean": 0.01207600224888854,
            "rv_p_value": 0.3434363961379136
          },
          "toward": {
            "alpha": 3.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": 0.3441101802285567,
            "rv_delta_ci_95": [
              0.0,
              0.01798810239605121
            ],
            "rv_delta_mean": 0.006110908655104675,
            "rv_p_value": 0.33211345618082977
          }
        },
        "overall": {
          "away": {
            "alpha": -3.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 24,
            "rv_cohens_dz": -0.16115950664034742,
            "rv_delta_ci_95": [
              -0.0679194609648818,
              0.018945258293685124
            ],
            "rv_delta_mean": -0.015580742374074592,
            "rv_p_value": 0.5158477003935452
          },
          "toward": {
            "alpha": 3.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 24,
            "rv_cohens_dz": 0.19706537621903525,
            "rv_delta_ci_95": [
              -0.022910829709996333,
              0.06787409235933906
            ],
            "rv_delta_mean": 0.01905972790192314,
            "rv_p_value": 0.457994792695103
          }
        },
        "recursive": {
          "away": {
            "alpha": -3.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": -0.39308744596183354,
            "rv_delta_ci_95": [
              -0.15916118603991114,
              0.016893948992062917
            ],
            "rv_delta_mean": -0.05509037754973621,
            "rv_p_value": 0.3384169003087605
          },
          "toward": {
            "alpha": 3.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": 0.24371430730796387,
            "rv_delta_ci_95": [
              -0.07186938195969987,
              0.15162221213097457
            ],
            "rv_delta_mean": 0.038482956772150835,
            "rv_p_value": 0.5765260030754056
          }
        }
      },
      "objective": {
        "score": -0.0006110908655104676,
        "score_breakdown": {
          "baseline_rv_penalty": 0.006110908655104675,
          "baseline_spillover_penalty": 0.0,
          "recursive_bt_gain": 0.0,
          "recursive_bt_suppression": -0.0,
          "recursive_rv_alignment": 0.0
        },
        "sign_checks": {
          "recursive_away_bt_negative": false,
          "recursive_away_rv_positive": false,
          "recursive_toward_bt_positive": false,
          "recursive_toward_rv_negative": false
        },
        "sign_match_count": 0
      },
      "rank": 8,
      "source_layer": 25,
      "state_source": {
        "centroid_cosine": 0.9391899108886719,
        "negative_centroid_norm": 216.47683715820312,
        "negative_selected_n": 52,
        "positive_centroid_norm": 226.54112243652344,
        "positive_selected_n": 72,
        "raw_direction_norm": 77.88246154785156
      },
      "window": 32
    },
    {
      "alpha": 2.0,
      "by_condition": {
        "away": {
          "alpha": -2.0,
          "by_prompt_mode": {
            "baseline": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "REPETITIVE": 1,
                "SURFACE": 11
              },
              "mean_generated_tokens": 73.66666666666667,
              "mean_output_rv": 1.1935958440382695,
              "n": 12,
              "std_output_rv": 0.2317858917413827
            },
            "recursive": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "SURFACE": 12
              },
              "mean_generated_tokens": 61.333333333333336,
              "mean_output_rv": 1.151546076339546,
              "n": 12,
              "std_output_rv": 0.1744551567911669
            }
          },
          "overall": {
            "bt_art_rate": 0.0,
            "class_counts": {
              "REPETITIVE": 1,
              "SURFACE": 23
            },
            "mean_generated_tokens": 67.5,
            "mean_output_rv": 1.175199070670078,
            "n": 24,
            "std_output_rv": 0.20320226830883134
          }
        },
        "none": {
          "alpha": 0.0,
          "by_prompt_mode": {
            "baseline": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "REPETITIVE": 1,
                "SURFACE": 11
              },
              "mean_generated_tokens": 81.08333333333333,
              "mean_output_rv": 1.1816740999209578,
              "n": 12,
              "std_output_rv": 0.22856714554613797
            },
            "recursive": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "SURFACE": 12
              },
              "mean_generated_tokens": 61.333333333333336,
              "mean_output_rv": 1.206636453889282,
              "n": 12,
              "std_output_rv": 0.19718131597462538
            }
          },
          "overall": {
            "bt_art_rate": 0.0,
            "class_counts": {
              "REPETITIVE": 1,
              "SURFACE": 23
            },
            "mean_generated_tokens": 71.20833333333333,
            "mean_output_rv": 1.1919527162608559,
            "n": 24,
            "std_output_rv": 0.21006473303167564
          }
        },
        "toward": {
          "alpha": 2.0,
          "by_prompt_mode": {
            "baseline": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "REPETITIVE": 1,
                "SURFACE": 11
              },
              "mean_generated_tokens": 81.08333333333333,
              "mean_output_rv": 1.1715727058237122,
              "n": 12,
              "std_output_rv": 0.1861512610894505
            },
            "recursive": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "SURFACE": 12
              },
              "mean_generated_tokens": 61.333333333333336,
              "mean_output_rv": 1.2358968297968143,
              "n": 12,
              "std_output_rv": 0.17719922557738418
            }
          },
          "overall": {
            "bt_art_rate": 0.0,
            "class_counts": {
              "REPETITIVE": 1,
              "SURFACE": 23
            },
            "mean_generated_tokens": 71.20833333333333,
            "mean_output_rv": 1.198059109812637,
            "n": 24,
            "std_output_rv": 0.1798098246509238
          }
        }
      },
      "candidate_name": "L23_W8_A2",
      "effects_by_prompt_mode": {
        "baseline": {
          "away": {
            "alpha": -2.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": 0.5153026085756349,
            "rv_delta_ci_95": [
              0.00242290910392103,
              0.03796583657971294
            ],
            "rv_delta_mean": 0.01654453523619205,
            "rv_p_value": 0.16071051258458285
          },
          "toward": {
            "alpha": 2.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": -0.11187859780303087,
            "rv_delta_ci_95": [
              -0.07107206113839462,
              0.03638308735283389
            ],
            "rv_delta_mean": -0.010101394097245553,
            "rv_p_value": 0.731641891676192
          }
        },
        "overall": {
          "away": {
            "alpha": -2.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 24,
            "rv_cohens_dz": -0.1498163067870397,
            "rv_delta_ci_95": [
              -0.0710966520745978,
              0.02095001522843139
            ],
            "rv_delta_mean": -0.014795739107651563,
            "rv_p_value": 0.5579394811524899
          },
          "toward": {
            "alpha": 2.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 24,
            "rv_cohens_dz": 0.05345858290199236,
            "rv_delta_ci_95": [
              -0.048534041051413956,
              0.05907459871067463
            ],
            "rv_delta_mean": 0.0061063935517805664,
            "rv_p_value": 0.8283355456049429
          }
        },
        "recursive": {
          "away": {
            "alpha": -2.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": -0.39308744596183354,
            "rv_delta_ci_95": [
              -0.15650579963490527,
              0.014204950100917728
            ],
            "rv_delta_mean": -0.05509037754973621,
            "rv_p_value": 0.3384169003087605
          },
          "toward": {
            "alpha": 2.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": 0.19954240382921784,
            "rv_delta_ci_95": [
              -0.06937843616543381,
              0.1335079212843053
            ],
            "rv_delta_mean": 0.029260375907532166,
            "rv_p_value": 0.6164873671243123
          }
        }
      },
      "objective": {
        "score": -0.0010101394097245555,
        "score_breakdown": {
          "baseline_rv_penalty": 0.010101394097245553,
          "baseline_spillover_penalty": 0.0,
          "recursive_bt_gain": 0.0,
          "recursive_bt_suppression": -0.0,
          "recursive_rv_alignment": 0.0
        },
        "sign_checks": {
          "recursive_away_bt_negative": false,
          "recursive_away_rv_positive": false,
          "recursive_toward_bt_positive": false,
          "recursive_toward_rv_negative": false
        },
        "sign_match_count": 0
      },
      "rank": 9,
      "source_layer": 23,
      "state_source": {
        "centroid_cosine": 0.8623574376106262,
        "negative_centroid_norm": 119.57177734375,
        "negative_selected_n": 57,
        "positive_centroid_norm": 130.5658416748047,
        "positive_selected_n": 72,
        "raw_direction_norm": 66.47270965576172
      },
      "window": 8
    },
    {
      "alpha": 2.0,
      "by_condition": {
        "away": {
          "alpha": -2.0,
          "by_prompt_mode": {
            "baseline": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "REPETITIVE": 1,
                "SURFACE": 11
              },
              "mean_generated_tokens": 73.66666666666667,
              "mean_output_rv": 1.1935958440382695,
              "n": 12,
              "std_output_rv": 0.2317858917413827
            },
            "recursive": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "SURFACE": 12
              },
              "mean_generated_tokens": 61.333333333333336,
              "mean_output_rv": 1.1474724692185319,
              "n": 12,
              "std_output_rv": 0.1848561696166616
            }
          },
          "overall": {
            "bt_art_rate": 0.0,
            "class_counts": {
              "REPETITIVE": 1,
              "SURFACE": 23
            },
            "mean_generated_tokens": 67.5,
            "mean_output_rv": 1.1734168675546344,
            "n": 24,
            "std_output_rv": 0.20707565767408906
          }
        },
        "none": {
          "alpha": 0.0,
          "by_prompt_mode": {
            "baseline": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "REPETITIVE": 1,
                "SURFACE": 11
              },
              "mean_generated_tokens": 81.08333333333333,
              "mean_output_rv": 1.1816740999209578,
              "n": 12,
              "std_output_rv": 0.22856714554613797
            },
            "recursive": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "SURFACE": 12
              },
              "mean_generated_tokens": 61.333333333333336,
              "mean_output_rv": 1.206636453889282,
              "n": 12,
              "std_output_rv": 0.19718131597462538
            }
          },
          "overall": {
            "bt_art_rate": 0.0,
            "class_counts": {
              "REPETITIVE": 1,
              "SURFACE": 23
            },
            "mean_generated_tokens": 71.20833333333333,
            "mean_output_rv": 1.1919527162608559,
            "n": 24,
            "std_output_rv": 0.21006473303167564
          }
        },
        "toward": {
          "alpha": 2.0,
          "by_prompt_mode": {
            "baseline": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "REPETITIVE": 1,
                "SURFACE": 11
              },
              "mean_generated_tokens": 81.08333333333333,
              "mean_output_rv": 1.1715727058237122,
              "n": 12,
              "std_output_rv": 0.1861512610894505
            },
            "recursive": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "SURFACE": 12
              },
              "mean_generated_tokens": 61.0,
              "mean_output_rv": 1.224100043858013,
              "n": 12,
              "std_output_rv": 0.15490370577520313
            }
          },
          "overall": {
            "bt_art_rate": 0.0,
            "class_counts": {
              "REPETITIVE": 1,
              "SURFACE": 23
            },
            "mean_generated_tokens": 71.04166666666667,
            "mean_output_rv": 1.1932016097201892,
            "n": 24,
            "std_output_rv": 0.17088056206874722
          }
        }
      },
      "candidate_name": "L23_W16_A2",
      "effects_by_prompt_mode": {
        "baseline": {
          "away": {
            "alpha": -2.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": 0.5153026085756349,
            "rv_delta_ci_95": [
              0.00242290910392103,
              0.040388745683633966
            ],
            "rv_delta_mean": 0.01654453523619205,
            "rv_p_value": 0.16071051258458285
          },
          "toward": {
            "alpha": 2.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": -0.11187859780303087,
            "rv_delta_ci_95": [
              -0.07107206113839462,
              0.03638308735283389
            ],
            "rv_delta_mean": -0.010101394097245553,
            "rv_p_value": 0.731641891676192
          }
        },
        "overall": {
          "away": {
            "alpha": -2.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 24,
            "rv_cohens_dz": -0.1717390508668533,
            "rv_delta_ci_95": [
              -0.0669719303882219,
              0.01863634673552487
            ],
            "rv_delta_mean": -0.016577942223095193,
            "rv_p_value": 0.5025934206823256
          },
          "toward": {
            "alpha": 2.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 24,
            "rv_cohens_dz": 0.01221246667899767,
            "rv_delta_ci_95": [
              -0.04356336117062587,
              0.049106028387268894
            ],
            "rv_delta_mean": 0.0012488934593329892,
            "rv_p_value": 0.9604641107081335
          }
        },
        "recursive": {
          "away": {
            "alpha": -2.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": -0.43904611101255947,
            "rv_delta_ci_95": [
              -0.16189029371535255,
              0.0026889988911451873
            ],
            "rv_delta_mean": -0.05916398467075022,
            "rv_p_value": 0.2895116630886113
          },
          "toward": {
            "alpha": 2.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": 0.14193760076900228,
            "rv_delta_ci_95": [
              -0.05215330539611354,
              0.10889221741901371
            ],
            "rv_delta_mean": 0.017463589968730906,
            "rv_p_value": 0.7201833065838955
          }
        }
      },
      "objective": {
        "score": -0.0010101394097245555,
        "score_breakdown": {
          "baseline_rv_penalty": 0.010101394097245553,
          "baseline_spillover_penalty": 0.0,
          "recursive_bt_gain": 0.0,
          "recursive_bt_suppression": -0.0,
          "recursive_rv_alignment": 0.0
        },
        "sign_checks": {
          "recursive_away_bt_negative": false,
          "recursive_away_rv_positive": false,
          "recursive_toward_bt_positive": false,
          "recursive_toward_rv_negative": false
        },
        "sign_match_count": 0
      },
      "rank": 10,
      "source_layer": 23,
      "state_source": {
        "centroid_cosine": 0.8739343881607056,
        "negative_centroid_norm": 120.42705535888672,
        "negative_selected_n": 57,
        "positive_centroid_norm": 130.48953247070312,
        "positive_selected_n": 72,
        "raw_direction_norm": 63.744468688964844
      },
      "window": 16
    },
    {
      "alpha": 2.0,
      "by_condition": {
        "away": {
          "alpha": -2.0,
          "by_prompt_mode": {
            "baseline": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "REPETITIVE": 1,
                "SURFACE": 11
              },
              "mean_generated_tokens": 73.66666666666667,
              "mean_output_rv": 1.1935958440382695,
              "n": 12,
              "std_output_rv": 0.2317858917413827
            },
            "recursive": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "SURFACE": 12
              },
              "mean_generated_tokens": 61.0,
              "mean_output_rv": 1.1545749442689908,
              "n": 12,
              "std_output_rv": 0.17747207771404513
            }
          },
          "overall": {
            "bt_art_rate": 0.0,
            "class_counts": {
              "REPETITIVE": 1,
              "SURFACE": 23
            },
            "mean_generated_tokens": 67.33333333333333,
            "mean_output_rv": 1.1765242003892098,
            "n": 24,
            "std_output_rv": 0.2040867438375433
          }
        },
        "none": {
          "alpha": 0.0,
          "by_prompt_mode": {
            "baseline": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "REPETITIVE": 1,
                "SURFACE": 11
              },
              "mean_generated_tokens": 81.08333333333333,
              "mean_output_rv": 1.1816740999209578,
              "n": 12,
              "std_output_rv": 0.22856714554613797
            },
            "recursive": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "SURFACE": 12
              },
              "mean_generated_tokens": 61.333333333333336,
              "mean_output_rv": 1.206636453889282,
              "n": 12,
              "std_output_rv": 0.19718131597462538
            }
          },
          "overall": {
            "bt_art_rate": 0.0,
            "class_counts": {
              "REPETITIVE": 1,
              "SURFACE": 23
            },
            "mean_generated_tokens": 71.20833333333333,
            "mean_output_rv": 1.1919527162608559,
            "n": 24,
            "std_output_rv": 0.21006473303167564
          }
        },
        "toward": {
          "alpha": 2.0,
          "by_prompt_mode": {
            "baseline": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "REPETITIVE": 1,
                "SURFACE": 11
              },
              "mean_generated_tokens": 81.08333333333333,
              "mean_output_rv": 1.1594967035748236,
              "n": 12,
              "std_output_rv": 0.17743267489538048
            },
            "recursive": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "SURFACE": 12
              },
              "mean_generated_tokens": 61.333333333333336,
              "mean_output_rv": 1.2120474808718404,
              "n": 12,
              "std_output_rv": 0.14600087584025528
            }
          },
          "overall": {
            "bt_art_rate": 0.0,
            "class_counts": {
              "REPETITIVE": 1,
              "SURFACE": 23
            },
            "mean_generated_tokens": 71.20833333333333,
            "mean_output_rv": 1.1811352589324189,
            "n": 24,
            "std_output_rv": 0.16252114297026085
          }
        }
      },
      "candidate_name": "L23_W32_A2",
      "effects_by_prompt_mode": {
        "baseline": {
          "away": {
            "alpha": -2.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": 0.5153026085756349,
            "rv_delta_ci_95": [
              0.0,
              0.03796583657971294
            ],
            "rv_delta_mean": 0.01654453523619205,
            "rv_p_value": 0.16071051258458285
          },
          "toward": {
            "alpha": 2.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": -0.2839840760846157,
            "rv_delta_ci_95": [
              -0.07295429567297011,
              0.006111945422231302
            ],
            "rv_delta_mean": -0.022177396346134092,
            "rv_p_value": 0.3925565830752674
          }
        },
        "overall": {
          "away": {
            "alpha": -2.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 24,
            "rv_cohens_dz": -0.13761008853207382,
            "rv_delta_ci_95": [
              -0.06373395236903022,
              0.02213655089003638
            ],
            "rv_delta_mean": -0.01347060938851944,
            "rv_p_value": 0.5901212767631678
          },
          "toward": {
            "alpha": 2.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 24,
            "rv_cohens_dz": -0.10661926434565494,
            "rv_delta_ci_95": [
              -0.057171437324931414,
              0.04012023170109977
            ],
            "rv_delta_mean": -0.010817457328437152,
            "rv_p_value": 0.666104806516041
          }
        },
        "recursive": {
          "away": {
            "alpha": -2.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": -0.3728898468699707,
            "rv_delta_ci_95": [
              -0.15624265501444912,
              0.016893948992062917
            ],
            "rv_delta_mean": -0.05206150962029136,
            "rv_p_value": 0.361949863823407
          },
          "toward": {
            "alpha": 2.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": 0.040580120053968854,
            "rv_delta_ci_95": [
              -0.07161201588658603,
              0.10650932133892287
            ],
            "rv_delta_mean": 0.00541102698255848,
            "rv_p_value": 0.9179995952044246
          }
        }
      },
      "objective": {
        "score": -0.0022177396346134094,
        "score_breakdown": {
          "baseline_rv_penalty": 0.022177396346134092,
          "baseline_spillover_penalty": 0.0,
          "recursive_bt_gain": 0.0,
          "recursive_bt_suppression": -0.0,
          "recursive_rv_alignment": 0.0
        },
        "sign_checks": {
          "recursive_away_bt_negative": false,
          "recursive_away_rv_positive": false,
          "recursive_toward_bt_positive": false,
          "recursive_toward_rv_negative": false
        },
        "sign_match_count": 0
      },
      "rank": 11,
      "source_layer": 23,
      "state_source": {
        "centroid_cosine": 0.8981044888496399,
        "negative_centroid_norm": 119.8793716430664,
        "negative_selected_n": 52,
        "positive_centroid_norm": 130.3688507080078,
        "positive_selected_n": 72,
        "raw_direction_norm": 57.402034759521484
      },
      "window": 32
    },
    {
      "alpha": 3.0,
      "by_condition": {
        "away": {
          "alpha": -3.0,
          "by_prompt_mode": {
            "baseline": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "REPETITIVE": 1,
                "SURFACE": 11
              },
              "mean_generated_tokens": 73.66666666666667,
              "mean_output_rv": 1.172580947832442,
              "n": 12,
              "std_output_rv": 0.18648847420439796
            },
            "recursive": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "SURFACE": 12
              },
              "mean_generated_tokens": 61.333333333333336,
              "mean_output_rv": 1.1539965993805978,
              "n": 12,
              "std_output_rv": 0.1784205952918939
            }
          },
          "overall": {
            "bt_art_rate": 0.0,
            "class_counts": {
              "REPETITIVE": 1,
              "SURFACE": 23
            },
            "mean_generated_tokens": 67.5,
            "mean_output_rv": 1.1644502953847602,
            "n": 24,
            "std_output_rv": 0.17712274217589244
          }
        },
        "none": {
          "alpha": 0.0,
          "by_prompt_mode": {
            "baseline": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "REPETITIVE": 1,
                "SURFACE": 11
              },
              "mean_generated_tokens": 81.08333333333333,
              "mean_output_rv": 1.1816740999209578,
              "n": 12,
              "std_output_rv": 0.22856714554613797
            },
            "recursive": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "SURFACE": 12
              },
              "mean_generated_tokens": 61.333333333333336,
              "mean_output_rv": 1.206636453889282,
              "n": 12,
              "std_output_rv": 0.19718131597462538
            }
          },
          "overall": {
            "bt_art_rate": 0.0,
            "class_counts": {
              "REPETITIVE": 1,
              "SURFACE": 23
            },
            "mean_generated_tokens": 71.20833333333333,
            "mean_output_rv": 1.1919527162608559,
            "n": 24,
            "std_output_rv": 0.21006473303167564
          }
        },
        "toward": {
          "alpha": 3.0,
          "by_prompt_mode": {
            "baseline": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "REPETITIVE": 1,
                "SURFACE": 11
              },
              "mean_generated_tokens": 73.58333333333333,
              "mean_output_rv": 1.1524097573063725,
              "n": 12,
              "std_output_rv": 0.18668856232295533
            },
            "recursive": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "SURFACE": 12
              },
              "mean_generated_tokens": 54.0,
              "mean_output_rv": 1.1949378123830448,
              "n": 12,
              "std_output_rv": 0.15053742973872727
            }
          },
          "overall": {
            "bt_art_rate": 0.0,
            "class_counts": {
              "REPETITIVE": 1,
              "SURFACE": 23
            },
            "mean_generated_tokens": 63.791666666666664,
            "mean_output_rv": 1.1694209793370414,
            "n": 24,
            "std_output_rv": 0.16874317570674907
          }
        }
      },
      "candidate_name": "L23_W8_A3",
      "effects_by_prompt_mode": {
        "baseline": {
          "away": {
            "alpha": -3.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": -0.05876085009293172,
            "rv_delta_ci_95": [
              -0.0538492319103691,
              0.03452468691720275
            ],
            "rv_delta_mean": -0.004470360969635218,
            "rv_p_value": 0.864452747075464
          },
          "toward": {
            "alpha": 3.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": -0.29898407757143947,
            "rv_delta_ci_95": [
              -0.07896895682043846,
              0.006791050469145891
            ],
            "rv_delta_mean": -0.024641551495704548,
            "rv_p_value": 0.3959335513415157
          }
        },
        "overall": {
          "away": {
            "alpha": -3.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 24,
            "rv_cohens_dz": -0.23598988777617977,
            "rv_delta_ci_95": [
              -0.08754201143942864,
              0.015961932091522954
            ],
            "rv_delta_mean": -0.025544514392969196,
            "rv_p_value": 0.360147658363853
          },
          "toward": {
            "alpha": 3.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 24,
            "rv_cohens_dz": -0.07187720287223485,
            "rv_delta_ci_95": [
              -0.06203459423828791,
              0.04851154696289988
            ],
            "rv_delta_mean": -0.007939649971334192,
            "rv_p_value": 0.7847906203803254
          }
        },
        "recursive": {
          "away": {
            "alpha": -3.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": -0.37173442292196274,
            "rv_delta_ci_95": [
              -0.15916118603991114,
              0.021307425151376593
            ],
            "rv_delta_mean": -0.05263985450868431,
            "rv_p_value": 0.36333458454177625
          },
          "toward": {
            "alpha": 3.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": 0.11526926800662729,
            "rv_delta_ci_95": [
              -0.08093596778682184,
              0.15075014397542447
            ],
            "rv_delta_mean": 0.017113202315221343,
            "rv_p_value": 0.7889891650742837
          }
        }
      },
      "objective": {
        "score": -0.002464155149570455,
        "score_breakdown": {
          "baseline_rv_penalty": 0.024641551495704548,
          "baseline_spillover_penalty": 0.0,
          "recursive_bt_gain": 0.0,
          "recursive_bt_suppression": -0.0,
          "recursive_rv_alignment": 0.0
        },
        "sign_checks": {
          "recursive_away_bt_negative": false,
          "recursive_away_rv_positive": false,
          "recursive_toward_bt_positive": false,
          "recursive_toward_rv_negative": false
        },
        "sign_match_count": 0
      },
      "rank": 12,
      "source_layer": 23,
      "state_source": {
        "centroid_cosine": 0.8623574376106262,
        "negative_centroid_norm": 119.57177734375,
        "negative_selected_n": 57,
        "positive_centroid_norm": 130.5658416748047,
        "positive_selected_n": 72,
        "raw_direction_norm": 66.47270965576172
      },
      "window": 8
    },
    {
      "alpha": 3.0,
      "by_condition": {
        "away": {
          "alpha": -3.0,
          "by_prompt_mode": {
            "baseline": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "REPETITIVE": 1,
                "SURFACE": 11
              },
              "mean_generated_tokens": 81.08333333333333,
              "mean_output_rv": 1.2185745886619883,
              "n": 12,
              "std_output_rv": 0.20327093666684912
            },
            "recursive": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "SURFACE": 12
              },
              "mean_generated_tokens": 60.833333333333336,
              "mean_output_rv": 1.164981516659196,
              "n": 12,
              "std_output_rv": 0.19359790950000433
            }
          },
          "overall": {
            "bt_art_rate": 0.0,
            "class_counts": {
              "REPETITIVE": 1,
              "SURFACE": 23
            },
            "mean_generated_tokens": 70.95833333333333,
            "mean_output_rv": 1.1965068531314267,
            "n": 24,
            "std_output_rv": 0.19502873890943526
          }
        },
        "none": {
          "alpha": 0.0,
          "by_prompt_mode": {
            "baseline": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "REPETITIVE": 1,
                "SURFACE": 11
              },
              "mean_generated_tokens": 81.08333333333333,
              "mean_output_rv": 1.1816740999209578,
              "n": 12,
              "std_output_rv": 0.22856714554613797
            },
            "recursive": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "SURFACE": 12
              },
              "mean_generated_tokens": 61.333333333333336,
              "mean_output_rv": 1.206636453889282,
              "n": 12,
              "std_output_rv": 0.19718131597462538
            }
          },
          "overall": {
            "bt_art_rate": 0.0,
            "class_counts": {
              "REPETITIVE": 1,
              "SURFACE": 23
            },
            "mean_generated_tokens": 71.20833333333333,
            "mean_output_rv": 1.1919527162608559,
            "n": 24,
            "std_output_rv": 0.21006473303167564
          }
        },
        "toward": {
          "alpha": 3.0,
          "by_prompt_mode": {
            "baseline": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "REPETITIVE": 1,
                "SURFACE": 11
              },
              "mean_generated_tokens": 73.58333333333333,
              "mean_output_rv": 1.1432212741999193,
              "n": 12,
              "std_output_rv": 0.18475582487533043
            },
            "recursive": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "SURFACE": 12
              },
              "mean_generated_tokens": 54.333333333333336,
              "mean_output_rv": 1.230406843135641,
              "n": 12,
              "std_output_rv": 0.18106481953096695
            }
          },
          "overall": {
            "bt_art_rate": 0.0,
            "class_counts": {
              "REPETITIVE": 1,
              "SURFACE": 23
            },
            "mean_generated_tokens": 63.958333333333336,
            "mean_output_rv": 1.1780955017742079,
            "n": 24,
            "std_output_rv": 0.1821233544512663
          }
        }
      },
      "candidate_name": "L21_W8_A3",
      "effects_by_prompt_mode": {
        "baseline": {
          "away": {
            "alpha": -3.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": 0.39680048997800843,
            "rv_delta_ci_95": [
              -0.004434680241826916,
              0.09786109898604359
            ],
            "rv_delta_mean": 0.03690048874103038,
            "rv_p_value": 0.24115823885751358
          },
          "toward": {
            "alpha": 3.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": -0.4157304787271051,
            "rv_delta_ci_95": [
              -0.08815743992689197,
              0.0003446235692628166
            ],
            "rv_delta_mean": -0.033830034602158054,
            "rv_p_value": 0.24760072494551738
          }
        },
        "overall": {
          "away": {
            "alpha": -3.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 24,
            "rv_cohens_dz": 0.036976408205989825,
            "rv_delta_ci_95": [
              -0.04661267210271228,
              0.05539962766894687
            ],
            "rv_delta_mean": 0.004554136870570517,
            "rv_p_value": 0.8807319970016081
          },
          "toward": {
            "alpha": 3.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 24,
            "rv_cohens_dz": 0.006399615196788373,
            "rv_delta_ci_95": [
              -0.04791241195482351,
              0.057446213572405605
            ],
            "rv_delta_mean": 0.0007348724658321778,
            "rv_p_value": 0.9805757969895637
          }
        },
        "recursive": {
          "away": {
            "alpha": -3.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": -0.27321844989129745,
            "rv_delta_ci_95": [
              -0.16708134127731933,
              0.05224515652916888
            ],
            "rv_delta_mean": -0.04165493723008643,
            "rv_p_value": 0.49695929629394014
          },
          "toward": {
            "alpha": 3.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": 0.3632866440237508,
            "rv_delta_ci_95": [
              -0.03743655187223383,
              0.17406122786196654
            ],
            "rv_delta_mean": 0.05258223306781753,
            "rv_p_value": 0.41429758800899813
          }
        }
      },
      "objective": {
        "score": -0.0033830034602158054,
        "score_breakdown": {
          "baseline_rv_penalty": 0.033830034602158054,
          "baseline_spillover_penalty": 0.0,
          "recursive_bt_gain": 0.0,
          "recursive_bt_suppression": -0.0,
          "recursive_rv_alignment": 0.0
        },
        "sign_checks": {
          "recursive_away_bt_negative": false,
          "recursive_away_rv_positive": false,
          "recursive_toward_bt_positive": false,
          "recursive_toward_rv_negative": false
        },
        "sign_match_count": 0
      },
      "rank": 13,
      "source_layer": 21,
      "state_source": {
        "centroid_cosine": 0.873100221157074,
        "negative_centroid_norm": 79.38467407226562,
        "negative_selected_n": 57,
        "positive_centroid_norm": 91.97340393066406,
        "positive_selected_n": 72,
        "raw_direction_norm": 44.85015869140625
      },
      "window": 8
    },
    {
      "alpha": 3.0,
      "by_condition": {
        "away": {
          "alpha": -3.0,
          "by_prompt_mode": {
            "baseline": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "REPETITIVE": 1,
                "SURFACE": 11
              },
              "mean_generated_tokens": 81.08333333333333,
              "mean_output_rv": 1.2183126617290623,
              "n": 12,
              "std_output_rv": 0.20351512818637157
            },
            "recursive": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "SURFACE": 12
              },
              "mean_generated_tokens": 60.833333333333336,
              "mean_output_rv": 1.1726297717463134,
              "n": 12,
              "std_output_rv": 0.20147105721811745
            }
          },
          "overall": {
            "bt_art_rate": 0.0,
            "class_counts": {
              "REPETITIVE": 1,
              "SURFACE": 23
            },
            "mean_generated_tokens": 70.95833333333333,
            "mean_output_rv": 1.1995020599714596,
            "n": 24,
            "std_output_rv": 0.19762692353459388
          }
        },
        "none": {
          "alpha": 0.0,
          "by_prompt_mode": {
            "baseline": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "REPETITIVE": 1,
                "SURFACE": 11
              },
              "mean_generated_tokens": 81.08333333333333,
              "mean_output_rv": 1.1816740999209578,
              "n": 12,
              "std_output_rv": 0.22856714554613797
            },
            "recursive": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "SURFACE": 12
              },
              "mean_generated_tokens": 61.333333333333336,
              "mean_output_rv": 1.206636453889282,
              "n": 12,
              "std_output_rv": 0.19718131597462538
            }
          },
          "overall": {
            "bt_art_rate": 0.0,
            "class_counts": {
              "REPETITIVE": 1,
              "SURFACE": 23
            },
            "mean_generated_tokens": 71.20833333333333,
            "mean_output_rv": 1.1919527162608559,
            "n": 24,
            "std_output_rv": 0.21006473303167564
          }
        },
        "toward": {
          "alpha": 3.0,
          "by_prompt_mode": {
            "baseline": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "REPETITIVE": 1,
                "SURFACE": 11
              },
              "mean_generated_tokens": 73.58333333333333,
              "mean_output_rv": 1.1432212741999193,
              "n": 12,
              "std_output_rv": 0.18475582487533043
            },
            "recursive": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "SURFACE": 12
              },
              "mean_generated_tokens": 54.333333333333336,
              "mean_output_rv": 1.228936857686615,
              "n": 12,
              "std_output_rv": 0.17753953004414297
            }
          },
          "overall": {
            "bt_art_rate": 0.0,
            "class_counts": {
              "REPETITIVE": 1,
              "SURFACE": 23
            },
            "mean_generated_tokens": 63.958333333333336,
            "mean_output_rv": 1.1775075075945975,
            "n": 24,
            "std_output_rv": 0.1806988265345048
          }
        }
      },
      "candidate_name": "L21_W16_A3",
      "effects_by_prompt_mode": {
        "baseline": {
          "away": {
            "alpha": -3.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": 0.3934800156169126,
            "rv_delta_ci_95": [
              -0.0058434297048134244,
              0.09890924571382415
            ],
            "rv_delta_mean": 0.03663856180810421,
            "rv_p_value": 0.2448202197442449
          },
          "toward": {
            "alpha": 3.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": -0.4157304787271051,
            "rv_delta_ci_95": [
              -0.09508223954363018,
              0.00017661957924718959
            ],
            "rv_delta_mean": -0.033830034602158054,
            "rv_p_value": 0.24760072494551738
          }
        },
        "overall": {
          "away": {
            "alpha": -3.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 24,
            "rv_cohens_dz": 0.061860660930930386,
            "rv_delta_ci_95": [
              -0.05618447852046291,
              0.060660867183649395
            ],
            "rv_delta_mean": 0.007549343710603569,
            "rv_p_value": 0.8019277982469991
          },
          "toward": {
            "alpha": 3.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 24,
            "rv_cohens_dz": 0.001180845875288252,
            "rv_delta_ci_95": [
              -0.05142502806044886,
              0.06979685293376711
            ],
            "rv_delta_mean": 0.00014687828622184096,
            "rv_p_value": 0.9964155000216341
          }
        },
        "recursive": {
          "away": {
            "alpha": -3.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": -0.222857311173927,
            "rv_delta_ci_95": [
              -0.14710240723315957,
              0.04527250897775803
            ],
            "rv_delta_mean": -0.03400668214296877,
            "rv_p_value": 0.5769549221864035
          },
          "toward": {
            "alpha": 3.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": 0.3080510688497342,
            "rv_delta_ci_95": [
              -0.052507081699979086,
              0.18685717365355117
            ],
            "rv_delta_mean": 0.051112247618791684,
            "rv_p_value": 0.48451016246181855
          }
        }
      },
      "objective": {
        "score": -0.0033830034602158054,
        "score_breakdown": {
          "baseline_rv_penalty": 0.033830034602158054,
          "baseline_spillover_penalty": 0.0,
          "recursive_bt_gain": 0.0,
          "recursive_bt_suppression": -0.0,
          "recursive_rv_alignment": 0.0
        },
        "sign_checks": {
          "recursive_away_bt_negative": false,
          "recursive_away_rv_positive": false,
          "recursive_toward_bt_positive": false,
          "recursive_toward_rv_negative": false
        },
        "sign_match_count": 0
      },
      "rank": 14,
      "source_layer": 21,
      "state_source": {
        "centroid_cosine": 0.8865320682525635,
        "negative_centroid_norm": 80.10830688476562,
        "negative_selected_n": 57,
        "positive_centroid_norm": 91.5705795288086,
        "positive_selected_n": 72,
        "raw_direction_norm": 42.38020324707031
      },
      "window": 16
    },
    {
      "alpha": 3.0,
      "by_condition": {
        "away": {
          "alpha": -3.0,
          "by_prompt_mode": {
            "baseline": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "REPETITIVE": 1,
                "SURFACE": 11
              },
              "mean_generated_tokens": 81.08333333333333,
              "mean_output_rv": 1.2166623201058773,
              "n": 12,
              "std_output_rv": 0.2033718414564915
            },
            "recursive": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "SURFACE": 12
              },
              "mean_generated_tokens": 60.833333333333336,
              "mean_output_rv": 1.165279889453345,
              "n": 12,
              "std_output_rv": 0.1924552882332221
            }
          },
          "overall": {
            "bt_art_rate": 0.0,
            "class_counts": {
              "REPETITIVE": 1,
              "SURFACE": 23
            },
            "mean_generated_tokens": 70.95833333333333,
            "mean_output_rv": 1.195504848660717,
            "n": 24,
            "std_output_rv": 0.19451002608853057
          }
        },
        "none": {
          "alpha": 0.0,
          "by_prompt_mode": {
            "baseline": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "REPETITIVE": 1,
                "SURFACE": 11
              },
              "mean_generated_tokens": 81.08333333333333,
              "mean_output_rv": 1.1816740999209578,
              "n": 12,
              "std_output_rv": 0.22856714554613797
            },
            "recursive": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "SURFACE": 12
              },
              "mean_generated_tokens": 61.333333333333336,
              "mean_output_rv": 1.206636453889282,
              "n": 12,
              "std_output_rv": 0.19718131597462538
            }
          },
          "overall": {
            "bt_art_rate": 0.0,
            "class_counts": {
              "REPETITIVE": 1,
              "SURFACE": 23
            },
            "mean_generated_tokens": 71.20833333333333,
            "mean_output_rv": 1.1919527162608559,
            "n": 24,
            "std_output_rv": 0.21006473303167564
          }
        },
        "toward": {
          "alpha": 3.0,
          "by_prompt_mode": {
            "baseline": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "REPETITIVE": 1,
                "SURFACE": 11
              },
              "mean_generated_tokens": 73.66666666666667,
              "mean_output_rv": 1.1432212741999193,
              "n": 12,
              "std_output_rv": 0.18475582487533043
            },
            "recursive": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "SURFACE": 12
              },
              "mean_generated_tokens": 55.0,
              "mean_output_rv": 1.212104236790957,
              "n": 12,
              "std_output_rv": 0.16913288268018992
            }
          },
          "overall": {
            "bt_art_rate": 0.0,
            "class_counts": {
              "REPETITIVE": 1,
              "SURFACE": 23
            },
            "mean_generated_tokens": 64.33333333333333,
            "mean_output_rv": 1.1707744592363343,
            "n": 24,
            "std_output_rv": 0.17590356200174218
          }
        }
      },
      "candidate_name": "L21_W32_A3",
      "effects_by_prompt_mode": {
        "baseline": {
          "away": {
            "alpha": -3.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": 0.3767783708065686,
            "rv_delta_ci_95": [
              0.0,
              0.09535292805971204
            ],
            "rv_delta_mean": 0.03498822018491945,
            "rv_p_value": 0.2639375616245001
          },
          "toward": {
            "alpha": 3.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": -0.4157304787271051,
            "rv_delta_ci_95": [
              -0.08798512814226056,
              0.0003446235692628166
            ],
            "rv_delta_mean": -0.033830034602158054,
            "rv_p_value": 0.24760072494551738
          }
        },
        "overall": {
          "away": {
            "alpha": -3.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 24,
            "rv_cohens_dz": 0.029137218023385345,
            "rv_delta_ci_95": [
              -0.05506832679722467,
              0.05761991604006246
            ],
            "rv_delta_mean": 0.0035521323998607543,
            "rv_p_value": 0.905871177466463
          },
          "toward": {
            "alpha": 3.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 24,
            "rv_cohens_dz": -0.05545484282560146,
            "rv_delta_ci_95": [
              -0.059336864111961186,
              0.05368077661641411
            ],
            "rv_delta_mean": -0.0065861700720413864,
            "rv_p_value": 0.8330392679167532
          }
        },
        "recursive": {
          "away": {
            "alpha": -3.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": -0.27450781368568766,
            "rv_delta_ci_95": [
              -0.15572794351312882,
              0.04143536402173896
            ],
            "rv_delta_mean": -0.04135656443593738,
            "rv_p_value": 0.49501238577860707
          },
          "toward": {
            "alpha": 3.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": 0.2144142917535866,
            "rv_delta_ci_95": [
              -0.07392484275897615,
              0.15584493437083552
            ],
            "rv_delta_mean": 0.034279626723133616,
            "rv_p_value": 0.6218883616416976
          }
        }
      },
      "objective": {
        "score": -0.0033830034602158054,
        "score_breakdown": {
          "baseline_rv_penalty": 0.033830034602158054,
          "baseline_spillover_penalty": 0.0,
          "recursive_bt_gain": 0.0,
          "recursive_bt_suppression": -0.0,
          "recursive_rv_alignment": 0.0
        },
        "sign_checks": {
          "recursive_away_bt_negative": false,
          "recursive_away_rv_positive": false,
          "recursive_toward_bt_positive": false,
          "recursive_toward_rv_negative": false
        },
        "sign_match_count": 0
      },
      "rank": 15,
      "source_layer": 21,
      "state_source": {
        "centroid_cosine": 0.9066073894500732,
        "negative_centroid_norm": 80.34516143798828,
        "negative_selected_n": 52,
        "positive_centroid_norm": 90.68419647216797,
        "positive_selected_n": 72,
        "raw_direction_norm": 38.3121452331543
      },
      "window": 32
    },
    {
      "alpha": 2.0,
      "by_condition": {
        "away": {
          "alpha": -2.0,
          "by_prompt_mode": {
            "baseline": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "REPETITIVE": 1,
                "SURFACE": 11
              },
              "mean_generated_tokens": 81.08333333333333,
              "mean_output_rv": 1.2021517543947673,
              "n": 12,
              "std_output_rv": 0.22019837547724064
            },
            "recursive": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "SURFACE": 12
              },
              "mean_generated_tokens": 60.833333333333336,
              "mean_output_rv": 1.1549730664804136,
              "n": 12,
              "std_output_rv": 0.17834067737432005
            }
          },
          "overall": {
            "bt_art_rate": 0.0,
            "class_counts": {
              "REPETITIVE": 1,
              "SURFACE": 23
            },
            "mean_generated_tokens": 70.95833333333333,
            "mean_output_rv": 1.1827252358417981,
            "n": 24,
            "std_output_rv": 0.19943411832984784
          }
        },
        "none": {
          "alpha": 0.0,
          "by_prompt_mode": {
            "baseline": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "REPETITIVE": 1,
                "SURFACE": 11
              },
              "mean_generated_tokens": 81.08333333333333,
              "mean_output_rv": 1.1816740999209578,
              "n": 12,
              "std_output_rv": 0.22856714554613797
            },
            "recursive": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "SURFACE": 12
              },
              "mean_generated_tokens": 61.333333333333336,
              "mean_output_rv": 1.206636453889282,
              "n": 12,
              "std_output_rv": 0.19718131597462538
            }
          },
          "overall": {
            "bt_art_rate": 0.0,
            "class_counts": {
              "REPETITIVE": 1,
              "SURFACE": 23
            },
            "mean_generated_tokens": 71.20833333333333,
            "mean_output_rv": 1.1919527162608559,
            "n": 24,
            "std_output_rv": 0.21006473303167564
          }
        },
        "toward": {
          "alpha": 2.0,
          "by_prompt_mode": {
            "baseline": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "REPETITIVE": 1,
                "SURFACE": 11
              },
              "mean_generated_tokens": 73.58333333333333,
              "mean_output_rv": 1.1407890947530759,
              "n": 12,
              "std_output_rv": 0.18492987395214538
            },
            "recursive": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "SURFACE": 12
              },
              "mean_generated_tokens": 54.333333333333336,
              "mean_output_rv": 1.2148860977039009,
              "n": 12,
              "std_output_rv": 0.17007631357379618
            }
          },
          "overall": {
            "bt_art_rate": 0.0,
            "class_counts": {
              "REPETITIVE": 1,
              "SURFACE": 23
            },
            "mean_generated_tokens": 63.958333333333336,
            "mean_output_rv": 1.1704278959334062,
            "n": 24,
            "std_output_rv": 0.17687516693188948
          }
        }
      },
      "candidate_name": "L21_W8_A2",
      "effects_by_prompt_mode": {
        "baseline": {
          "away": {
            "alpha": -2.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": 0.6257330525954974,
            "rv_delta_ci_95": [
              0.003069877914459518,
              0.0422745205297552
            ],
            "rv_delta_mean": 0.020477654473809382,
            "rv_p_value": 0.07921991222458594
          },
          "toward": {
            "alpha": 2.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": -0.4387436060980264,
            "rv_delta_ci_95": [
              -0.09058961937373529,
              0.0003446235692628166
            ],
            "rv_delta_mean": -0.03626221404900138,
            "rv_p_value": 0.22455637939822992
          }
        },
        "overall": {
          "away": {
            "alpha": -2.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 24,
            "rv_cohens_dz": -0.0938421648795141,
            "rv_delta_ci_95": [
              -0.062374840684229556,
              0.02700238173098301
            ],
            "rv_delta_mean": -0.009227480419058102,
            "rv_p_value": 0.7039123573469961
          },
          "toward": {
            "alpha": 2.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 24,
            "rv_cohens_dz": -0.058666554002557554,
            "rv_delta_ci_95": [
              -0.06321007468971103,
              0.05109814420217614
            ],
            "rv_delta_mean": -0.006932733374969824,
            "rv_p_value": 0.8235413657074102
          }
        },
        "recursive": {
          "away": {
            "alpha": -2.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": -0.35990997284290605,
            "rv_delta_ci_95": [
              -0.16366422882574197,
              0.0230598952855271
            ],
            "rv_delta_mean": -0.05166338740886879,
            "rv_p_value": 0.37774696989228573
          },
          "toward": {
            "alpha": 2.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": 0.23774873101258676,
            "rv_delta_ci_95": [
              -0.06285158139937683,
              0.16322805700026852
            ],
            "rv_delta_mean": 0.03706148763607751,
            "rv_p_value": 0.5855936849467147
          }
        }
      },
      "objective": {
        "score": -0.0036262214049001384,
        "score_breakdown": {
          "baseline_rv_penalty": 0.03626221404900138,
          "baseline_spillover_penalty": 0.0,
          "recursive_bt_gain": 0.0,
          "recursive_bt_suppression": -0.0,
          "recursive_rv_alignment": 0.0
        },
        "sign_checks": {
          "recursive_away_bt_negative": false,
          "recursive_away_rv_positive": false,
          "recursive_toward_bt_positive": false,
          "recursive_toward_rv_negative": false
        },
        "sign_match_count": 0
      },
      "rank": 16,
      "source_layer": 21,
      "state_source": {
        "centroid_cosine": 0.873100221157074,
        "negative_centroid_norm": 79.38467407226562,
        "negative_selected_n": 57,
        "positive_centroid_norm": 91.97340393066406,
        "positive_selected_n": 72,
        "raw_direction_norm": 44.85015869140625
      },
      "window": 8
    },
    {
      "alpha": 2.0,
      "by_condition": {
        "away": {
          "alpha": -2.0,
          "by_prompt_mode": {
            "baseline": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "REPETITIVE": 1,
                "SURFACE": 11
              },
              "mean_generated_tokens": 81.08333333333333,
              "mean_output_rv": 1.1869245960289463,
              "n": 12,
              "std_output_rv": 0.23199040218090058
            },
            "recursive": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "SURFACE": 12
              },
              "mean_generated_tokens": 60.916666666666664,
              "mean_output_rv": 1.1549730664804136,
              "n": 12,
              "std_output_rv": 0.17834067737432005
            }
          },
          "overall": {
            "bt_art_rate": 0.0,
            "class_counts": {
              "REPETITIVE": 1,
              "SURFACE": 23
            },
            "mean_generated_tokens": 71.0,
            "mean_output_rv": 1.1737680838619033,
            "n": 24,
            "std_output_rv": 0.2060661317464464
          }
        },
        "none": {
          "alpha": 0.0,
          "by_prompt_mode": {
            "baseline": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "REPETITIVE": 1,
                "SURFACE": 11
              },
              "mean_generated_tokens": 81.08333333333333,
              "mean_output_rv": 1.1816740999209578,
              "n": 12,
              "std_output_rv": 0.22856714554613797
            },
            "recursive": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "SURFACE": 12
              },
              "mean_generated_tokens": 61.333333333333336,
              "mean_output_rv": 1.206636453889282,
              "n": 12,
              "std_output_rv": 0.19718131597462538
            }
          },
          "overall": {
            "bt_art_rate": 0.0,
            "class_counts": {
              "REPETITIVE": 1,
              "SURFACE": 23
            },
            "mean_generated_tokens": 71.20833333333333,
            "mean_output_rv": 1.1919527162608559,
            "n": 24,
            "std_output_rv": 0.21006473303167564
          }
        },
        "toward": {
          "alpha": 2.0,
          "by_prompt_mode": {
            "baseline": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "REPETITIVE": 1,
                "SURFACE": 11
              },
              "mean_generated_tokens": 73.58333333333333,
              "mean_output_rv": 1.1407890947530759,
              "n": 12,
              "std_output_rv": 0.18492987395214538
            },
            "recursive": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "SURFACE": 12
              },
              "mean_generated_tokens": 54.333333333333336,
              "mean_output_rv": 1.2148860977039009,
              "n": 12,
              "std_output_rv": 0.17007631357379618
            }
          },
          "overall": {
            "bt_art_rate": 0.0,
            "class_counts": {
              "REPETITIVE": 1,
              "SURFACE": 23
            },
            "mean_generated_tokens": 63.958333333333336,
            "mean_output_rv": 1.1704278959334062,
            "n": 24,
            "std_output_rv": 0.17687516693188948
          }
        }
      },
      "candidate_name": "L21_W16_A2",
      "effects_by_prompt_mode": {
        "baseline": {
          "away": {
            "alpha": -2.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": 0.46605962793602546,
            "rv_delta_ci_95": [
              0.0,
              0.012681610409505818
            ],
            "rv_delta_mean": 0.005250496107988445,
            "rv_p_value": 0.17462808606243144
          },
          "toward": {
            "alpha": 2.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": -0.4387436060980264,
            "rv_delta_ci_95": [
              -0.09058961937373529,
              0.0003446235692628166
            ],
            "rv_delta_mean": -0.03626221404900138,
            "rv_p_value": 0.22455637939822992
          }
        },
        "overall": {
          "away": {
            "alpha": -2.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 24,
            "rv_cohens_dz": -0.19572640015242726,
            "rv_delta_ci_95": [
              -0.06417357803290281,
              0.013472289753584728
            ],
            "rv_delta_mean": -0.01818463239895277,
            "rv_p_value": 0.431499527793695
          },
          "toward": {
            "alpha": 2.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 24,
            "rv_cohens_dz": -0.058666554002557554,
            "rv_delta_ci_95": [
              -0.06072078826666766,
              0.0532631262635567
            ],
            "rv_delta_mean": -0.006932733374969824,
            "rv_p_value": 0.8235413657074102
          }
        },
        "recursive": {
          "away": {
            "alpha": -2.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": -0.35990997284290605,
            "rv_delta_ci_95": [
              -0.16626366109037,
              0.02316466805618051
            ],
            "rv_delta_mean": -0.05166338740886879,
            "rv_p_value": 0.37774696989228573
          },
          "toward": {
            "alpha": 2.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": 0.23774873101258676,
            "rv_delta_ci_95": [
              -0.06535415287801799,
              0.153126915815316
            ],
            "rv_delta_mean": 0.03706148763607751,
            "rv_p_value": 0.5855936849467147
          }
        }
      },
      "objective": {
        "score": -0.0036262214049001384,
        "score_breakdown": {
          "baseline_rv_penalty": 0.03626221404900138,
          "baseline_spillover_penalty": 0.0,
          "recursive_bt_gain": 0.0,
          "recursive_bt_suppression": -0.0,
          "recursive_rv_alignment": 0.0
        },
        "sign_checks": {
          "recursive_away_bt_negative": false,
          "recursive_away_rv_positive": false,
          "recursive_toward_bt_positive": false,
          "recursive_toward_rv_negative": false
        },
        "sign_match_count": 0
      },
      "rank": 17,
      "source_layer": 21,
      "state_source": {
        "centroid_cosine": 0.8865320682525635,
        "negative_centroid_norm": 80.10830688476562,
        "negative_selected_n": 57,
        "positive_centroid_norm": 91.5705795288086,
        "positive_selected_n": 72,
        "raw_direction_norm": 42.38020324707031
      },
      "window": 16
    },
    {
      "alpha": 3.0,
      "by_condition": {
        "away": {
          "alpha": -3.0,
          "by_prompt_mode": {
            "baseline": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "REPETITIVE": 1,
                "SURFACE": 11
              },
              "mean_generated_tokens": 73.66666666666667,
              "mean_output_rv": 1.1714620007388854,
              "n": 12,
              "std_output_rv": 0.1900568959144823
            },
            "recursive": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "SURFACE": 12
              },
              "mean_generated_tokens": 61.333333333333336,
              "mean_output_rv": 1.1539965993805978,
              "n": 12,
              "std_output_rv": 0.1784205952918939
            }
          },
          "overall": {
            "bt_art_rate": 0.0,
            "class_counts": {
              "REPETITIVE": 1,
              "SURFACE": 23
            },
            "mean_generated_tokens": 67.5,
            "mean_output_rv": 1.1638208876446348,
            "n": 24,
            "std_output_rv": 0.17910472511656525
          }
        },
        "none": {
          "alpha": 0.0,
          "by_prompt_mode": {
            "baseline": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "REPETITIVE": 1,
                "SURFACE": 11
              },
              "mean_generated_tokens": 81.08333333333333,
              "mean_output_rv": 1.1816740999209578,
              "n": 12,
              "std_output_rv": 0.22856714554613797
            },
            "recursive": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "SURFACE": 12
              },
              "mean_generated_tokens": 61.333333333333336,
              "mean_output_rv": 1.206636453889282,
              "n": 12,
              "std_output_rv": 0.19718131597462538
            }
          },
          "overall": {
            "bt_art_rate": 0.0,
            "class_counts": {
              "REPETITIVE": 1,
              "SURFACE": 23
            },
            "mean_generated_tokens": 71.20833333333333,
            "mean_output_rv": 1.1919527162608559,
            "n": 24,
            "std_output_rv": 0.21006473303167564
          }
        },
        "toward": {
          "alpha": 3.0,
          "by_prompt_mode": {
            "baseline": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "REPETITIVE": 1,
                "SURFACE": 11
              },
              "mean_generated_tokens": 73.58333333333333,
              "mean_output_rv": 1.1407890947530759,
              "n": 12,
              "std_output_rv": 0.18492987395214538
            },
            "recursive": {
              "bt_art_rate": 0.0,
              "class_counts": {
                "SURFACE": 12
              },
              "mean_generated_tokens": 61.0,
              "mean_output_rv": 1.2329636338666747,
              "n": 12,
              "std_output_rv": 0.17695378530682862
            }
          },
          "overall": {
            "bt_art_rate": 0.0,
            "class_counts": {
              "REPETITIVE": 1,
              "SURFACE": 23
            },
            "mean_generated_tokens": 67.29166666666667,
            "mean_output_rv": 1.1811154556152754,
            "n": 24,
            "std_output_rv": 0.18164468844157625
          }
        }
      },
      "candidate_name": "L23_W16_A3",
      "effects_by_prompt_mode": {
        "baseline": {
          "away": {
            "alpha": -3.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": -0.09295899454607003,
            "rv_delta_ci_95": [
              -0.04639536281785878,
              0.02845678621786142
            ],
            "rv_delta_mean": -0.005589308063191739,
            "rv_p_value": 0.7874125424797528
          },
          "toward": {
            "alpha": 3.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": -0.4387436060980264,
            "rv_delta_ci_95": [
              -0.09994659843731683,
              0.0003446235692628166
            ],
            "rv_delta_mean": -0.03626221404900138,
            "rv_p_value": 0.22455637939822992
          }
        },
        "overall": {
          "away": {
            "alpha": -3.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 24,
            "rv_cohens_dz": -0.2550655806655203,
            "rv_delta_ci_95": [
              -0.0771249935772025,
              0.015267976421821941
            ],
            "rv_delta_mean": -0.02617392213309474,
            "rv_p_value": 0.32377834761645363
          },
          "toward": {
            "alpha": 3.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 24,
            "rv_cohens_dz": -0.07584296770245674,
            "rv_delta_ci_95": [
              -0.05908382306040202,
              0.0510823514083402
            ],
            "rv_delta_mean": -0.008879354162453994,
            "rv_p_value": 0.7657736062611472
          }
        },
        "recursive": {
          "away": {
            "alpha": -3.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": -0.37173442292196274,
            "rv_delta_ci_95": [
              -0.15916118603991114,
              0.01799997258754222
            ],
            "rv_delta_mean": -0.05263985450868431,
            "rv_p_value": 0.36333458454177625
          },
          "toward": {
            "alpha": 3.0,
            "bt_art_cohens_h": 0.0,
            "bt_art_exact_sign_p": null,
            "bt_art_prompt_losses": 0,
            "bt_art_prompt_wins": 0,
            "bt_art_rate_control": 0.0,
            "bt_art_rate_delta": 0.0,
            "bt_art_rate_delta_ci_95": [
              0.0,
              0.0
            ],
            "bt_art_rate_treated": 0.0,
            "n_prompt_pairs": 12,
            "rv_cohens_dz": 0.1751730242442123,
            "rv_delta_ci_95": [
              -0.07410965877786047,
              0.1393766430122142
            ],
            "rv_delta_mean": 0.026327179977392645,
            "rv_p_value": 0.6593674066578894
          }
        }
      },
      "objective": {
        "score": -0.0036262214049001384,
        "score_breakdown": {
          "baseline_rv_penalty": 0.03626221404900138,
          "baseline_spillover_penalty": 0.0,
          "recursive_bt_gain": 0.0,
          "recursive_bt_suppression": -0.0,
          "recursive_rv_alignment": 0.0
        },
        "sign_checks": {
          "recursive_away_bt_negative": false,
          "recursive_away_rv_positive": false,
          "recursive_toward_bt_positive": false,
          "recursive_toward_rv_negative": false
        },
        "sign_match_count": 0
      },
      "rank": 18,
      "source_layer": 23,
      "state_source": {
        "centroid_cosine": 0.8739343881607056,
        "negative_centroid_norm": 120.42705535888672,
        "negative_selected_n": 57,
        "positive_centroid_norm": 130.48953247070312,
        "positive_selected_n": 72,
        "raw_direction_norm": 63.744468688964844
      },
      "window": 16
    }
  ],
  "device": "cuda",
  "experiment": "causal_state_targeted_scan_v1",
  "generation_seeds": [
    111
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
  "model": "Qwen/Qwen2.5-7B",
  "model_name": "Qwen/Qwen2.5-7B",
  "n_generation_seeds": 1,
  "promotion_recommendation": {
    "alpha": 2.0,
    "interventions": [
      {
        "alpha": -2.0,
        "name": "away_alpha_best"
      },
      {
        "alpha": 0.0,
        "name": "none"
      },
      {
        "alpha": 2.0,
        "name": "toward_alpha_best"
      }
    ],
    "source_layer": 21,
    "window": 32
  },
  "prompt_bank_version": "2ac959a313614329",
  "schema_version": "metrics_summary_v1",
  "search_space": {
    "candidate_alpha_magnitudes": [
      2.0,
      3.0
    ],
    "candidate_count": 18,
    "candidate_source_layers": [
      21,
      23,
      25
    ],
    "candidate_windows": [
      8,
      16,
      32
    ]
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
  "timestamp": "20260315_064222",
  "verdict": "search_completed"
}
```
