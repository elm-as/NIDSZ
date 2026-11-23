"""Main orchestration entrypoint for the NIDSZ intrusion detection pipeline."""

from pathlib import Path

from src.core.config import NidszConfig, resolve_project_paths
from src.core.contracts import NidszReportPayload
from src.evaluation.shap_behavioral_audit import (
    ShapComparisonInput,
    compute_shap_behavioral_audit,
)
from src.features.flow_preprocessor import build_flow_feature_matrix
from src.ingestion.flow_schema_validator import load_and_validate_flows
from src.modeling.holdout_benchmark import run_holdout_benchmark_suite
from src.modeling.zero_day_loao_runner import (
    LoaoBenchmarkInput,
    run_zero_day_loao_benchmark,
)
from src.visualization.results_exporter import export_results_json
from src.visualization.soc_dashboard import build_soc_dashboard_html
from src.visualization.soc_plots import render_all_soc_figures


def _print_summary(payload: NidszReportPayload) -> None:
    report = payload.schema_report
    print(
        f"[NIDSZ] Flows audited: {report.total_rows:,} "
        f"({report.train_rows:,} train / {report.test_rows:,} holdout test)"
    )
    print(
        f"[NIDSZ] TTL cols present: {report.ttl_columns_present} | "
        f"Testbed shortcuts ablated: {len(report.testbed_shortcuts_present)}"
    )
    print(
        f"[NIDSZ] Cross-split exact feature overlap: "
        f"{report.cross_split_overlap_test_rows:,} test rows "
        f"({report.cross_split_overlap_rate:.2%})"
    )
    print(
        f"[NIDSZ] Best Full: {payload.best_full_model} ({payload.best_full_calibration}) | "
        f"Best Ablated: {payload.best_ablated_model} ({payload.best_ablated_calibration})"
    )


def run_nidsz_pipeline(project_root: Path | None = None) -> NidszReportPayload:
    """Execute ingestion, ablation benchmarks, LOAO zero-day study, SHAP, and visuals."""
    paths = resolve_project_paths(project_root)
    config = NidszConfig()

    # 1- Importation et controle des flux reseau officiels
    print("[1/6] Loading and auditing official UNSW-NB15 parquet partitions...")
    loaded = load_and_validate_flows(paths.train_parquet, paths.test_parquet)

    # 2- Preparation des variables completes et sans raccourcis lab
    print("[2/6] Building Full-34 and Shortcut-Ablated-24 feature matrices...")
    full_matrix = build_flow_feature_matrix(
        loaded.train_df, loaded.test_df, "full_flow"
    )
    ablated_matrix = build_flow_feature_matrix(
        loaded.train_df, loaded.test_df, "shortcut_ablated"
    )

    # 3- Entrainement holdout et calibration anti-fatigue SOC
    print("[3/6] Running official holdout benchmarks + 3-fold OOF calibration...")
    full_suite = run_holdout_benchmark_suite(full_matrix, config)
    ablated_suite = run_holdout_benchmark_suite(ablated_matrix, config)

    # 4- Evaluation zero-day (familles d'attaques inconnues) et SHAP
    print("[4/6] Running 7-family Leave-One-Attack-Family-Out (Zero-Day LOAO)...")
    loao_results = run_zero_day_loao_benchmark(
        LoaoBenchmarkInput(
            train_df=loaded.train_df,
            test_df=loaded.test_df,
            iso_detector_full=full_suite.fitted_isolation_forest,
            config=config,
        )
    )

    print("[5/6] Computing TreeSHAP behavioral telemetry attribution...")
    shap_summary = compute_shap_behavioral_audit(
        ShapComparisonInput(
            full_model=full_suite.fitted_lightgbm,
            full_x_test=full_matrix.x_test,
            ablated_model=ablated_suite.fitted_lightgbm,
            ablated_x_test=ablated_matrix.x_test,
            sample_size=config.shap_sample_size,
            random_seed=config.random_seed,
        )
    )

    payload = NidszReportPayload(
        schema_report=loaded.schema_report,
        full_artifacts=full_suite.artifacts,
        ablated_artifacts=ablated_suite.artifacts,
        loao_results=loao_results,
        shap_summary=shap_summary,
        best_full_model=full_suite.best_model_name,
        best_full_calibration=full_suite.best_calibration_method,
        best_ablated_model=ablated_suite.best_model_name,
        best_ablated_calibration=ablated_suite.best_calibration_method,
        random_seed=config.random_seed,
    )

    # 5- Export des graphiques, du dashboard et des resultats JSON
    print("[6/6] Exporting JSON, 4 publication 300 DPI PNGs, and HTML5 dashboard...")
    export_results_json(payload, paths.results_json)
    render_all_soc_figures(payload, paths.images_dir)
    build_soc_dashboard_html(payload, paths.dashboard_html)
    _print_summary(payload)
    return payload


if __name__ == "__main__":
    run_nidsz_pipeline()
