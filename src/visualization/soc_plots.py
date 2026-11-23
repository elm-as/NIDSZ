"""Publication-grade 300 DPI cybersecurity SOC figures for NIDSZ."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.core.contracts import ModelHoldoutArtifact, NidszReportPayload

_PALETTE = {
    "slate": "#0F172A",
    "crimson": "#BE123C",
    "teal": "#0D9488",
    "cobalt": "#1D4ED8",
    "amber": "#B45309",
    "muted": "#64748B",
    "grid": "#E2E8F0",
    "border": "#CBD5E1",
}

_MODEL_STYLES = {
    "lightgbm": ("LightGBM", "#BE123C"),
    "hist_gradient_boosting": ("HistGB", "#1D4ED8"),
    "logistic_regression": ("Logistic Reg.", "#0D9488"),
    "isolation_forest": ("IsolationForest (Normal-Only)", "#B45309"),
}


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": _PALETTE["border"],
            "axes.grid": True,
            "grid.color": _PALETTE["grid"],
            "grid.linewidth": 0.65,
            "font.size": 9.3,
        }
    )


def _find_artifact(
    artifacts: tuple[ModelHoldoutArtifact, ...], model_name: str, cal_method: str
) -> ModelHoldoutArtifact:
    return next(
        a
        for a in artifacts
        if a.model_name == model_name and a.calibration_method == cal_method
    )


def _primary_model_artifacts(
    artifacts: tuple[ModelHoldoutArtifact, ...],
) -> list[ModelHoldoutArtifact]:
    selected: list[ModelHoldoutArtifact] = []
    for name in _MODEL_STYLES:
        cal = "unsupervised_normal_only" if name == "isolation_forest" else "platt"
        selected.append(_find_artifact(artifacts, name, cal))
    return selected


def _plot_soc_roc_low_fpr(payload: NidszReportPayload, output_path: Path) -> Path:
    fig, (ax_roc, ax_bar) = plt.subplots(1, 2, figsize=(13.0, 5.2))
    primaries = _primary_model_artifacts(payload.full_artifacts)

    for art in primaries:
        label, color = _MODEL_STYLES[art.model_name]
        fpr = np.maximum(np.asarray(art.roc_curve_fpr), 1e-4)
        tpr = np.asarray(art.roc_curve_tpr)
        ax_roc.plot(
            fpr, tpr, lw=2.0, color=color,
            label=f"{label} (AUC={art.metrics.roc_auc:.4f})",
        )

    for budget, ls, y_pos in ((0.001, "--", 0.94), (0.005, "-.", 0.94), (0.01, ":", 0.94)):
        ax_roc.axvline(budget, color=_PALETTE["slate"], linestyle=ls, lw=1.1, alpha=0.75)
        ax_roc.text(budget * 1.08, y_pos, f"{budget * 100:.1f}%", fontsize=8, fontweight="bold", color=_PALETTE["slate"])

    ax_roc.set_xscale("log")
    ax_roc.set_xlim(1e-4, 1.0)
    ax_roc.set_ylim(0.0, 1.03)
    ax_roc.set_xlabel("False Positive Rate (Log Scale — SOC Alert Budget)")
    ax_roc.set_ylabel("True Positive Rate (Detection Recall)")
    ax_roc.set_title("Low-FPR SOC ROC Curve (Official Holdout, n=82,332)", fontweight="bold")
    ax_roc.legend(loc="lower right", frameon=True, facecolor="#FFFFFF", fontsize=8.0)

    x_pos = np.arange(len(primaries))
    width = 0.25
    b_colors = (_PALETTE["crimson"], _PALETTE["cobalt"], _PALETTE["teal"])
    b_labels = ("FPR <= 0.1%", "FPR <= 0.5%", "FPR <= 1.0%")
    for idx in range(3):
        vals = [art.soc_operating_points[idx].recall_tpr * 100.0 for art in primaries]
        bars = ax_bar.bar(x_pos + (idx - 1) * width, vals, width=width, color=b_colors[idx], label=b_labels[idx])
        for bar, val in zip(bars, vals):
            ax_bar.text(bar.get_x() + bar.get_width() / 2, val + 1.2, f"{val:.1f}%", ha="center", fontsize=7.5)

    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels([_MODEL_STYLES[a.model_name][0].replace(" (Normal-Only)", "\n(Unsup.)") for a in primaries], fontsize=8.5)
    ax_bar.set_ylim(0, 105)
    ax_bar.set_ylabel("Attack Detection Recall (%)")
    ax_bar.set_title("Alert-Fatigue Recall Under Strict SOC FPR Ceilings", fontweight="bold")
    ax_bar.legend(frameon=True, facecolor="#FFFFFF", fontsize=8.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_zero_day_loao(payload: NidszReportPayload, output_path: Path) -> Path:
    fig, (ax_rec, ax_delta) = plt.subplots(1, 2, figsize=(13.4, 5.4))
    rows = sorted(payload.loao_results, key=lambda r: r.ablated_recall_at_010, reverse=True)
    labels = [f"{r.family_name}\n(n={r.test_attack_count:,})" for r in rows]
    y_pos = np.arange(len(rows))
    h = 0.26

    full_r = [r.full_recall_at_010 * 100.0 for r in rows]
    abl_r = [r.ablated_recall_at_010 * 100.0 for r in rows]
    iso_r = [r.iso_recall_at_010 * 100.0 for r in rows]

    ax_rec.barh(y_pos - h, full_r, height=h, color=_PALETTE["crimson"], label="LGBM Full (34 vars)")
    ax_rec.barh(y_pos, abl_r, height=h, color=_PALETTE["teal"], label="LGBM Behavioral-24 (Ablated)")
    ax_rec.barh(y_pos + h, iso_r, height=h, color=_PALETTE["amber"], label="IsolationForest (Normal-Only)")
    ax_rec.set_yticks(y_pos)
    ax_rec.set_yticklabels(labels, fontsize=8.5)
    ax_rec.invert_yaxis()
    ax_rec.set_xlim(0, 108)
    ax_rec.set_xlabel("Unseen Zero-Day Detection Recall @ FPR <= 1.0% (%)")
    ax_rec.set_title("Leave-One-Attack-Family-Out (Zero-Day Recall @ 1% FPR)", fontweight="bold")
    ax_rec.legend(loc="lower right", frameon=True, facecolor="#FFFFFF", fontsize=8.0)

    deltas = [(r.ablated_recall_at_001 - r.full_recall_at_001) * 100.0 for r in rows]
    d_colors = [_PALETTE["teal"] if d >= 0 else _PALETTE["crimson"] for d in deltas]
    ax_delta.barh(y_pos, deltas, height=0.55, color=d_colors)
    ax_delta.axvline(0.0, color=_PALETTE["slate"], lw=1.2)
    for idx, val in enumerate(deltas):
        offset = 0.6 if val >= 0 else -0.6
        ha = "left" if val >= 0 else "right"
        ax_delta.text(val + offset, idx, f"{val:+.1f} pp", va="center", ha=ha, fontsize=8.2, fontweight="bold")

    ax_delta.set_yticks(y_pos)
    ax_delta.set_yticklabels([r.family_name for r in rows], fontsize=8.5)
    ax_delta.invert_yaxis()
    max_abs = max(abs(v) for v in deltas) + 4.5
    ax_delta.set_xlim(-max_abs, max_abs)
    ax_delta.set_xlabel("Behavioral-24 minus Full-34 Recall Delta (pp) @ FPR <= 0.1%")
    ax_delta.set_title("Shortcut Ablation Impact on Strict Zero-Day Detection", fontweight="bold")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_ablation_and_shap(payload: NidszReportPayload, output_path: Path) -> Path:
    fig, (ax_ab, ax_shap) = plt.subplots(1, 2, figsize=(13.4, 5.4))
    full_lgb = _find_artifact(payload.full_artifacts, "lightgbm", "platt")
    abl_lgb = _find_artifact(payload.ablated_artifacts, "lightgbm", "platt")

    metric_labels = ("ROC-AUC", "PR-AUC", "Recall @0.1%", "Recall @0.5%", "Recall @1.0%")
    full_vals = [
        full_lgb.metrics.roc_auc * 100.0,
        full_lgb.metrics.pr_auc * 100.0,
        full_lgb.soc_operating_points[0].recall_tpr * 100.0,
        full_lgb.soc_operating_points[1].recall_tpr * 100.0,
        full_lgb.soc_operating_points[2].recall_tpr * 100.0,
    ]
    abl_vals = [
        abl_lgb.metrics.roc_auc * 100.0,
        abl_lgb.metrics.pr_auc * 100.0,
        abl_lgb.soc_operating_points[0].recall_tpr * 100.0,
        abl_lgb.soc_operating_points[1].recall_tpr * 100.0,
        abl_lgb.soc_operating_points[2].recall_tpr * 100.0,
    ]
    x_pos = np.arange(len(metric_labels))
    w = 0.35
    ax_ab.bar(x_pos - w / 2, full_vals, width=w, color=_PALETTE["crimson"], label="Full Flow (34 vars)")
    ax_ab.bar(x_pos + w / 2, abl_vals, width=w, color=_PALETTE["teal"], label="Shortcut-Ablated (24 vars)")
    ax_ab.set_xticks(x_pos)
    ax_ab.set_xticklabels(metric_labels, fontsize=8.5)
    ax_ab.set_ylim(72, 102)
    ax_ab.set_ylabel("Holdout Score (%)")
    ttl_note = "TTL Pre-Pruned (0/3)" if not payload.schema_report.ttl_columns_present else "TTL Dropped"
    ax_ab.set_title(f"TTL & Testbed Shortcut Ablation ({ttl_note}, 10 Shortcuts Dropped)", fontweight="bold")
    ax_ab.legend(frameon=True, facecolor="#FFFFFF", fontsize=8.3)

    top_shap = payload.shap_summary.full_top_features[:12]
    names = [item.feature_name for item in top_shap]
    vals = [item.mean_abs_shap for item in top_shap]
    group_colors = {
        "TTL / Testbed Shortcut": _PALETTE["crimson"],
        "Protocol / Service": _PALETTE["amber"],
        "Timing / Jitter / RTT": _PALETTE["cobalt"],
        "Volumetric / Packet": _PALETTE["teal"],
    }
    colors = [group_colors.get(item.feature_group, _PALETTE["slate"]) for item in top_shap]
    y_pos = np.arange(len(names))
    ax_shap.barh(y_pos, vals, color=colors)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=col, label=grp)
        for grp, col in group_colors.items()
    ]
    ax_shap.legend(handles=handles, loc="lower right", frameon=True, facecolor="#FFFFFF", fontsize=7.8)
    ax_shap.set_yticks(y_pos)
    ax_shap.set_yticklabels(names, fontsize=8.5)
    ax_shap.invert_yaxis()
    ax_shap.set_xlabel("Mean |SHAP Value| (Log-Odds Contribution)")
    ax_shap.set_title(
        f"TreeSHAP Top 12 Features (Shortcut Share: {payload.shap_summary.shortcut_shap_share:.1%})",
        fontweight="bold",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_calibration_and_latency(payload: NidszReportPayload, output_path: Path) -> Path:
    fig, (ax_cal, ax_lat) = plt.subplots(1, 2, figsize=(13.0, 5.2))
    ax_cal.plot([0, 1], [0, 1], linestyle="--", color=_PALETTE["slate"], lw=1.3, label="Ideal y = x")
    cal_curves = (
        ("raw", _PALETTE["crimson"], "o", "Raw LightGBM"),
        ("platt", _PALETTE["cobalt"], "s", "Platt (OOF Sigmoid)"),
        ("isotonic", _PALETTE["teal"], "^", "Isotonic (OOF)"),
    )
    for method, color, marker, label in cal_curves:
        art = _find_artifact(payload.full_artifacts, "lightgbm", method)
        bins = [b for b in art.reliability_bins_10 if b.sample_count > 0]
        ax_cal.plot(
            [b.mean_predicted for b in bins],
            [b.empirical_rate for b in bins],
            marker=marker, color=color, lw=1.9, markersize=5,
            label=f"{label} (ECE={art.metrics.ece_uniform_10:.4f})",
        )
    ax_cal.set_xlim(0, 1)
    ax_cal.set_ylim(0, 1)
    ax_cal.set_xlabel("Mean Predicted Attack Probability")
    ax_cal.set_ylabel("Empirical Attack Rate (Official Holdout)")
    ax_cal.set_title("Probability Calibration Reliability (Prior-Shift Audit)", fontweight="bold")
    ax_cal.legend(loc="upper left", frameon=True, facecolor="#FFFFFF", fontsize=8.0)

    primaries = _primary_model_artifacts(payload.full_artifacts)
    for art in primaries:
        label, color = _MODEL_STYLES[art.model_name]
        tp = art.latency.throughput_flows_per_sec
        rec = art.soc_operating_points[0].recall_tpr * 100.0
        ax_lat.scatter([tp], [rec], s=140, color=color, edgecolors="#0F172A", linewidth=0.9, zorder=4)
        ax_lat.annotate(
            f"{label.split(' ')[0]}\n({art.latency.latency_us_per_flow:.1f} us/flow)",
            xy=(tp, rec), xytext=(8, -14 if rec > 70 else 8),
            textcoords="offset points", fontsize=8.2, color=_PALETTE["slate"], fontweight="bold",
        )

    ax_lat.set_xscale("log")
    min_tp = min(a.latency.throughput_flows_per_sec for a in primaries)
    max_tp = max(a.latency.throughput_flows_per_sec for a in primaries)
    ax_lat.set_xlim(min_tp * 0.65, max_tp * 2.1)
    ax_lat.set_ylim(-5, 102)
    ax_lat.set_xlabel("Batch Inference Throughput (Flows / Second, Log Scale)")
    ax_lat.set_ylabel("Recall @ Strict FPR <= 0.1% (%)")
    ax_lat.set_title("SOC Line-Rate Pareto: Detection Speed vs Low-FPR Recall", fontweight="bold")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_all_soc_figures(
    payload: NidszReportPayload, images_dir: Path
) -> tuple[Path, ...]:
    """Render all four 300 DPI NIDSZ figures into `images_dir`."""
    images_dir.mkdir(parents=True, exist_ok=True)
    _configure_style()
    return (
        _plot_soc_roc_low_fpr(payload, images_dir / "nidsz_soc_roc_low_fpr.png"),
        _plot_zero_day_loao(payload, images_dir / "nidsz_zero_day_loao_recall.png"),
        _plot_ablation_and_shap(payload, images_dir / "nidsz_ttl_ablation_shap.png"),
        _plot_calibration_and_latency(
            payload, images_dir / "nidsz_calibration_and_latency.png"
        ),
    )
