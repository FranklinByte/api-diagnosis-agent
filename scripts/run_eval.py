from __future__ import annotations

import json
import os
from pathlib import Path

from api_diagnosis.evaluator import run_benchmark


if __name__ == "__main__":
    result = run_benchmark()
    output_path = os.getenv("OFFLINE_EVAL_OUTPUT")
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_keys = (
        "total",
        "unique_input_count",
        "status_accuracy",
        "category_accuracy",
        "category_macro_accuracy",
        "tool_call_rate",
        "tool_selection_accuracy",
        "citation_coverage",
        "mean_rounds",
        "oracle_derived_rate",
        "oracle_fallback_rate",
        "prediction_source_counts",
        "tool_usage_counts",
    )
    summary = {
        name: {key: metrics[key] for key in summary_keys}
        for name, metrics in result.items()
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
