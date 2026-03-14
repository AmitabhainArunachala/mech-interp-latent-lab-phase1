from __future__ import annotations

import numpy as np


PERSISTED_AGGREGATE_KEYS = (
    "n_sessions",
    "total_turns",
    "total_bt_art",
    "bt_art_rate",
    "mean_rv",
    "std_rv",
    "n_rv",
    "n_rv_missing",
    "rv_missing_rate",
    "total_malformed",
    "malformed_rate",
    "total_repetitive",
    "repetitive_rate",
    "mean_alpha_ratio",
)


def summarize_session_metrics(session):
    """Normalize per-session counts so aggregate serialization has one source of truth."""
    turns = session.get("turns", [])
    total_turns = int(session.get("max_turns", len(turns)))
    total_bt_art = int(session.get("bt_art_count", 0))
    rv_values = [turn.get("output_rv") for turn in turns]
    valid_rvs = [float(value) for value in rv_values if value is not None]
    alpha_values = [turn.get("alpha_ratio") for turn in turns if turn.get("alpha_ratio") is not None]
    malformed_count = int(session.get(
        "malformed_count",
        sum(turn.get("classification") == "MALFORMED" for turn in turns),
    ))
    repetitive_count = int(session.get(
        "repetitive_count",
        sum(turn.get("classification") == "REPETITIVE" for turn in turns),
    ))
    mean_rv = session.get("mean_rv")
    if mean_rv is None and valid_rvs:
        mean_rv = float(np.mean(valid_rvs))
    std_rv = session.get("std_rv")
    if std_rv is None and valid_rvs:
        std_rv = float(np.std(valid_rvs))
    mean_alpha_ratio = session.get("mean_alpha_ratio")
    if mean_alpha_ratio is None and alpha_values:
        mean_alpha_ratio = float(np.mean(alpha_values))

    return {
        "id": session["session_id"],
        "total_turns": total_turns,
        "bt_art": total_bt_art,
        "rate": float(session.get("bt_art_rate", total_bt_art / total_turns if total_turns else 0.0)),
        "mean_rv": mean_rv,
        "std_rv": std_rv,
        "n_rv": len(valid_rvs),
        "n_rv_missing": sum(value is None for value in rv_values),
        "mean_alpha_ratio": mean_alpha_ratio,
        "malformed_count": malformed_count,
        "repetitive_count": repetitive_count,
        "dist": session.get("classification_dist", {}),
    }


def aggregate_sessions(sessions):
    """Build the aggregate block persisted in the result JSON."""
    session_summaries = [summarize_session_metrics(session) for session in sessions]
    total_turns = sum(summary["total_turns"] for summary in session_summaries)
    total_bt_art = sum(summary["bt_art"] for summary in session_summaries)
    total_malformed = sum(summary["malformed_count"] for summary in session_summaries)
    total_repetitive = sum(summary["repetitive_count"] for summary in session_summaries)

    all_rvs = []
    all_alpha = []
    for session, summary in zip(sessions, session_summaries):
        turns = session.get("turns", [])
        if turns:
            all_rvs.extend(
                float(turn["output_rv"])
                for turn in turns
                if turn.get("output_rv") is not None
            )
            all_alpha.extend(
                float(turn["alpha_ratio"])
                for turn in turns
                if turn.get("alpha_ratio") is not None
            )
        elif summary["mean_alpha_ratio"] is not None and summary["total_turns"] > 0:
            all_alpha.extend([float(summary["mean_alpha_ratio"])] * summary["total_turns"])

    total_rv_missing = sum(summary["n_rv_missing"] for summary in session_summaries)
    return {
        "n_sessions": len(sessions),
        "total_turns": total_turns,
        "total_bt_art": total_bt_art,
        "bt_art_rate": total_bt_art / total_turns if total_turns > 0 else 0,
        "mean_rv": float(np.mean(all_rvs)) if all_rvs else None,
        "std_rv": float(np.std(all_rvs)) if all_rvs else None,
        "n_rv": len(all_rvs),
        "n_rv_missing": int(total_rv_missing),
        "rv_missing_rate": float(total_rv_missing / total_turns) if total_turns > 0 else 0.0,
        "total_malformed": int(total_malformed),
        "malformed_rate": float(total_malformed / total_turns) if total_turns > 0 else 0.0,
        "total_repetitive": int(total_repetitive),
        "repetitive_rate": float(total_repetitive / total_turns) if total_turns > 0 else 0.0,
        "mean_alpha_ratio": float(np.mean(all_alpha)) if all_alpha else None,
        "per_session": session_summaries,
    }


def serialize_aggregates(aggregates):
    """Persist aggregate stats without per-session detail while retaining required keys."""
    serialized = {}
    for prefix, summary in aggregates.items():
        persisted = {key: summary.get(key) for key in PERSISTED_AGGREGATE_KEYS}
        for key, value in summary.items():
            if key == "per_session" or key in persisted:
                continue
            persisted[key] = value
        serialized[prefix] = persisted
    return serialized
