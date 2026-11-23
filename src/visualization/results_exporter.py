"""JSON serialization of NIDSZ benchmark, ablation, LOAO, and SHAP results."""

from dataclasses import asdict
import json
from pathlib import Path

from src.core.contracts import ModelHoldoutArtifact, NidszReportPayload


def _serialize_artifact(art: ModelHoldoutArtifact) -> dict[str, object]:
    raw = asdict(art)
    raw["roc_curve_fpr"] = [round(v, 6) for v in art.roc_curve_fpr]
    raw["roc_curve_tpr"] = [round(v, 6) for v in art.roc_curve_tpr]
    return raw


def serialize_nidsz_payload(payload: NidszReportPayload) -> dict[str, object]:
    """Convert `NidszReportPayload` into a pure JSON-serializable dictionary."""
    return {
        "schema_report": asdict(payload.schema_report),
        "best_full_model": payload.best_full_model,
        "best_full_calibration": payload.best_full_calibration,
        "best_ablated_model": payload.best_ablated_model,
        "best_ablated_calibration": payload.best_ablated_calibration,
        "random_seed": payload.random_seed,
        "full_artifacts": [_serialize_artifact(a) for a in payload.full_artifacts],
        "ablated_artifacts": [
            _serialize_artifact(a) for a in payload.ablated_artifacts
        ],
        "loao_results": [asdict(item) for item in payload.loao_results],
        "shap_summary": asdict(payload.shap_summary),
    }


def export_results_json(payload: NidszReportPayload, output_path: Path) -> Path:
    """Write complete verified NIDSZ results to `output_path` in UTF-8 JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = serialize_nidsz_payload(payload)
    output_path.write_text(
        json.dumps(serialized, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return output_path
