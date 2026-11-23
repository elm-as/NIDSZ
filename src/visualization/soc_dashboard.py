"""Standalone light editorial HTML5 + Plotly SOC intrusion dashboard."""

import json
from pathlib import Path

from src.core.contracts import NidszReportPayload
from src.visualization.results_exporter import serialize_nidsz_payload

_DASHBOARD_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: #F8FAFC; color: #0F172A; line-height: 1.45; padding: 20px;
}
.container { max-width: 1380px; margin: 0 auto; }
.header {
  background: #FFFFFF; border: 1px solid #CBD5E1; border-left: 5px solid #BE123C;
  border-radius: 6px; padding: 18px 22px; margin-bottom: 16px;
  display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;
}
.header h1 { font-size: 1.28rem; font-weight: 700; color: #0F172A; }
.header p { font-size: 0.87rem; color: #475569; margin-top: 3px; }
.meta-strip { display: flex; gap: 8px; flex-wrap: wrap; }
.meta-tag {
  font-size: 0.78rem; padding: 4px 10px; border-radius: 4px;
  border: 1px solid #CBD5E1; background: #F8FAFC; color: #0F172A; font-weight: 600;
}
.kpi-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(205px, 1fr));
  gap: 12px; margin-bottom: 16px;
}
.kpi-card { background: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 6px; padding: 14px 16px; }
.kpi-label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.04em; color: #475569; font-weight: 600; }
.kpi-val {
  font-size: 1.45rem; font-weight: 700; color: #0F172A; margin: 4px 0;
  font-variant-numeric: tabular-nums; font-feature-settings: "tnum" 1;
  font-family: "JetBrains Mono", "Consolas", ui-monospace, monospace;
}
.kpi-sub { font-size: 0.77rem; color: #475569; }
.accent-crimson { color: #BE123C; }
.accent-teal { color: #0D9488; }
.accent-cobalt { color: #1D4ED8; }
.accent-amber { color: #B45309; }
.chart-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(560px, 1fr));
  gap: 14px; margin-bottom: 16px;
}
.panel { background: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 6px; padding: 14px 16px; }
.panel-head {
  display: flex; justify-content: space-between; align-items: center;
  margin-bottom: 10px; gap: 10px; flex-wrap: wrap;
}
.panel-title { font-size: 0.95rem; font-weight: 700; color: #0F172A; }
select {
  border: 1px solid #CBD5E1; border-radius: 4px; padding: 4px 8px;
  font-size: 0.82rem; background: #FFFFFF; color: #0F172A;
}
.table-wrap { overflow-x: auto; margin-top: 8px; }
table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
th, td { padding: 7px 10px; border-bottom: 1px solid #E2E8F0; text-align: left; }
th { background: #F8FAFC; color: #334155; font-weight: 600; border-top: 1px solid #CBD5E1; }
.num, td.num, th.num {
  text-align: right; font-variant-numeric: tabular-nums; font-feature-settings: "tnum" 1;
  font-family: "JetBrains Mono", "Consolas", ui-monospace, monospace;
}
"""

_DASHBOARD_JS = """
const DATA = __NIDSZ_JSON__;
const layoutBase = {
  template: "plotly_white", paper_bgcolor: "#FFFFFF", plot_bgcolor: "#FFFFFF",
  margin: { l: 58, r: 22, t: 24, b: 48 }, font: { family: "Inter, sans-serif", color: "#0F172A", size: 11 }
};
const modelColors = {
  lightgbm: "#BE123C", hist_gradient_boosting: "#1D4ED8",
  logistic_regression: "#0D9488", isolation_forest: "#B45309"
};

function pickPrimary(artifacts, modelName) {
  const cal = modelName === "isolation_forest" ? "unsupervised_normal_only" : "platt";
  return artifacts.find(a => a.model_name === modelName && a.calibration_method === cal);
}

function renderLowFprRoc() {
  const mode = document.getElementById("roc-mode").value;
  const list = mode === "full_flow" ? DATA.full_artifacts : DATA.ablated_artifacts;
  const models = ["lightgbm", "hist_gradient_boosting", "logistic_regression", "isolation_forest"];
  const traces = models.map(m => {
    const a = pickPrimary(list, m);
    return {
      x: a.roc_curve_fpr.map(v => Math.max(v, 1e-4)), y: a.roc_curve_tpr, mode: "lines",
      name: `${m} (AUC=${a.metrics.roc_auc.toFixed(4)})`, line: { color: modelColors[m], width: 2.1 }
    };
  });
  Plotly.newPlot("plot-roc", traces, {
    ...layoutBase,
    xaxis: { title: "False Positive Rate (Log Scale)", type: "log", range: [-4, 0], gridcolor: "#E2E8F0" },
    yaxis: { title: "True Positive Rate (Recall)", range: [0, 1.02], gridcolor: "#E2E8F0" },
    shapes: [0.001, 0.005, 0.01].map(b => ({ type: "line", x0: b, x1: b, y0: 0, y1: 1, line: { color: "#0F172A", dash: "dot", width: 1 } })),
    legend: { orientation: "h", y: 1.15 }
  }, { responsive: true });
}

function renderLoao() {
  const budgetKey = document.getElementById("loao-budget").value;
  const rows = [...DATA.loao_results].sort((a, b) => b["ablated_" + budgetKey] - a["ablated_" + budgetKey]);
  const fams = rows.map(r => r.family_name);
  const traces = [
    { x: fams, y: rows.map(r => r["full_" + budgetKey] * 100), name: "LGBM Full (34 vars)", type: "bar", marker: { color: "#BE123C" } },
    { x: fams, y: rows.map(r => r["ablated_" + budgetKey] * 100), name: "LGBM Behavioral-24", type: "bar", marker: { color: "#0D9488" } },
    { x: fams, y: rows.map(r => r["iso_" + budgetKey] * 100), name: "IsolationForest (Unsup.)", type: "bar", marker: { color: "#B45309" } }
  ];
  Plotly.newPlot("plot-loao", traces, {
    ...layoutBase, barmode: "group",
    yaxis: { title: "Zero-Day Detection Recall (%)", range: [0, 105], gridcolor: "#E2E8F0" },
    legend: { orientation: "h", y: 1.15 }
  }, { responsive: true });
}

function renderShap() {
  const mode = document.getElementById("shap-mode").value;
  const items = (mode === "full" ? DATA.shap_summary.full_top_features : DATA.shap_summary.ablated_top_features).slice(0, 12).reverse();
  const groupPalette = {
    "TTL / Testbed Shortcut": "#BE123C", "Protocol / Service": "#B45309",
    "Timing / Jitter / RTT": "#1D4ED8", "Volumetric / Packet": "#0D9488"
  };
  Plotly.newPlot("plot-shap", [{
    y: items.map(i => i.feature_name), x: items.map(i => i.mean_abs_shap),
    type: "bar", orientation: "h", marker: { color: items.map(i => groupPalette[i.feature_group] || "#0F172A") },
    text: items.map(i => i.feature_group), hovertemplate: "%{y}: %{x:.4f} (%{text})<extra></extra>"
  }], { ...layoutBase, margin: { l: 120, r: 22, t: 20, b: 46 }, xaxis: { title: "Mean |SHAP| (Log-Odds)", gridcolor: "#E2E8F0" } }, { responsive: true });
}

function renderCalibration() {
  const pick = (m) => DATA.full_artifacts.find(a => a.model_name === "lightgbm" && a.calibration_method === m);
  const traces = [{ x: [0, 1], y: [0, 1], mode: "lines", name: "Ideal y = x", line: { dash: "dash", color: "#0F172A", width: 1.4 } }];
  [["raw", "#BE123C", "Raw LGBM"], ["platt", "#1D4ED8", "Platt OOF"], ["isotonic", "#0D9488", "Isotonic OOF"]].forEach(([m, color, label]) => {
    const bins = pick(m).reliability_bins_10.filter(b => b.sample_count > 0);
    traces.push({
      x: bins.map(b => b.mean_predicted), y: bins.map(b => b.empirical_rate), mode: "lines+markers",
      name: `${label} (ECE=${pick(m).metrics.ece_uniform_10.toFixed(4)})`, line: { color, width: 2 }
    });
  });
  Plotly.newPlot("plot-cal", traces, {
    ...layoutBase, xaxis: { title: "Mean Predicted Attack Probability", range: [0, 1], gridcolor: "#E2E8F0" },
    yaxis: { title: "Empirical Holdout Attack Rate", range: [0, 1], gridcolor: "#E2E8F0" },
    legend: { orientation: "h", y: 1.15 }
  }, { responsive: true });
}

function populateTables() {
  const allArts = [...DATA.full_artifacts, ...DATA.ablated_artifacts].filter(
    a => a.calibration_method === "platt" || a.calibration_method === "unsupervised_normal_only" || (a.model_name === "lightgbm" && a.calibration_method === "isotonic")
  );
  document.getElementById("holdout-table-body").innerHTML = allArts.map(a => `<tr>
    <td>${a.feature_mode}</td><td>${a.model_name}</td><td>${a.calibration_method}</td>
    <td class="num">${a.metrics.roc_auc.toFixed(4)}</td><td class="num">${a.metrics.pr_auc.toFixed(4)}</td>
    <td class="num">${a.metrics.f1_optimal.toFixed(4)}</td><td class="num">${a.metrics.brier_score.toFixed(4)}</td>
    <td class="num">${a.metrics.ece_uniform_10.toFixed(4)}</td>
    <td class="num">${(a.soc_operating_points[0].recall_tpr * 100).toFixed(2)}%</td>
    <td class="num">${(a.soc_operating_points[2].recall_tpr * 100).toFixed(2)}%</td>
    <td class="num">${a.latency.latency_us_per_flow.toFixed(2)}</td>
  </tr>`).join("");

  document.getElementById("loao-table-body").innerHTML = DATA.loao_results.map(r => `<tr>
    <td>${r.family_name}</td><td class="num">${r.test_attack_count.toLocaleString()}</td>
    <td class="num">${r.full_roc_auc.toFixed(4)}</td><td class="num">${r.ablated_roc_auc.toFixed(4)}</td>
    <td class="num">${r.iso_forest_roc_auc.toFixed(4)}</td>
    <td class="num">${(r.full_recall_at_001 * 100).toFixed(2)}%</td>
    <td class="num">${(r.ablated_recall_at_001 * 100).toFixed(2)}%</td>
    <td class="num">${(r.full_recall_at_010 * 100).toFixed(2)}%</td>
    <td class="num">${(r.ablated_recall_at_010 * 100).toFixed(2)}%</td>
    <td class="num">${(r.iso_recall_at_010 * 100).toFixed(2)}%</td>
  </tr>`).join("");
}

renderLowFprRoc(); renderLoao(); renderShap(); renderCalibration(); populateTables();
document.getElementById("roc-mode").addEventListener("change", renderLowFprRoc);
document.getElementById("loao-budget").addEventListener("change", renderLoao);
document.getElementById("shap-mode").addEventListener("change", renderShap);
"""


def _extract_kpis(serialized: dict[str, object]) -> dict[str, str]:
    ds = serialized["schema_report"]
    full_lgb = next(
        a for a in serialized["full_artifacts"]
        if a["model_name"] == "lightgbm" and a["calibration_method"] == "platt"
    )
    abl_lgb = next(
        a for a in serialized["ablated_artifacts"]
        if a["model_name"] == "lightgbm" and a["calibration_method"] == "platt"
    )
    loao = serialized["loao_results"]
    mean_loao_abl = sum(r["ablated_recall_at_010"] for r in loao) / len(loao)
    mean_loao_full = sum(r["full_recall_at_010"] for r in loao) / len(loao)
    return {
        "total_flows": f"{ds['total_rows']:,}",
        "split_sub": f"{ds['train_rows']:,} train / {ds['test_rows']:,} holdout",
        "full_auc": f"{full_lgb['metrics']['roc_auc']:.4f}",
        "full_prauc": f"{full_lgb['metrics']['pr_auc']:.4f}",
        "rec_001": f"{full_lgb['soc_operating_points'][0]['recall_tpr'] * 100:.2f}%",
        "rec_010": f"{full_lgb['soc_operating_points'][2]['recall_tpr'] * 100:.2f}%",
        "abl_rec_001": f"{abl_lgb['soc_operating_points'][0]['recall_tpr'] * 100:.2f}%",
        "abl_auc": f"{abl_lgb['metrics']['roc_auc']:.4f}",
        "loao_mean": f"{mean_loao_abl * 100:.2f}%",
        "loao_delta": f"{(mean_loao_abl - mean_loao_full) * 100:+.2f} pp vs Full",
        "throughput": f"{full_lgb['latency']['throughput_flows_per_sec']:,.0f}",
        "us_flow": f"{full_lgb['latency']['latency_us_per_flow']:.2f} us/flow",
    }


def build_soc_dashboard_html(
    payload: NidszReportPayload, dashboard_path: Path
) -> Path:
    """Build a standalone light editorial HTML5 Plotly SOC dashboard."""
    dashboard_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = serialize_nidsz_payload(payload)
    kpis = _extract_kpis(serialized)
    json_blob = json.dumps(serialized)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>NIDSZ - Network Intrusion Detection &amp; Zero-Day SOC Monitor</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>{_DASHBOARD_CSS}</style>
</head>
<body>
<div class="container">
  <header class="header">
    <div>
      <h1>NIDSZ - Network Intrusion Detection &amp; Zero-Day Generalization Monitor</h1>
      <p>UNSW-NB15 Official Partition (257,673 Flows) | TTL &amp; Testbed Shortcut Ablation, Low-FPR SOC Budgets &amp; 7-Family LOAO</p>
    </div>
    <div class="meta-strip">
      <span class="meta-tag">Official Train/Test Split</span>
      <span class="meta-tag">Seed 42 | 3-Fold OOF Cal.</span>
      <span class="meta-tag">ELMAS.md Compliant</span>
    </div>
  </header>

  <section class="kpi-grid">
    <div class="kpi-card"><div class="kpi-label">Network Flows Audited</div><div class="kpi-val">{kpis['total_flows']}</div><div class="kpi-sub">{kpis['split_sub']}</div></div>
    <div class="kpi-card"><div class="kpi-label">Holdout ROC-AUC (LGBM)</div><div class="kpi-val accent-teal">{kpis['full_auc']}</div><div class="kpi-sub">PR-AUC: {kpis['full_prauc']}</div></div>
    <div class="kpi-card"><div class="kpi-label">Recall @ FPR &lt;= 0.1%</div><div class="kpi-val accent-crimson">{kpis['rec_001']}</div><div class="kpi-sub">Recall @ 1.0% FPR: {kpis['rec_010']}</div></div>
    <div class="kpi-card"><div class="kpi-label">Behavioral-24 Recall @0.1%</div><div class="kpi-val accent-teal">{kpis['abl_rec_001']}</div><div class="kpi-sub">Ablated ROC-AUC: {kpis['abl_auc']}</div></div>
    <div class="kpi-card"><div class="kpi-label">Mean Zero-Day LOAO @1%</div><div class="kpi-val accent-amber">{kpis['loao_mean']}</div><div class="kpi-sub">{kpis['loao_delta']}</div></div>
    <div class="kpi-card"><div class="kpi-label">SOC Batch Throughput</div><div class="kpi-val accent-cobalt">{kpis['throughput']}</div><div class="kpi-sub">flows/sec ({kpis['us_flow']})</div></div>
  </section>

  <section class="chart-grid">
    <div class="panel">
      <div class="panel-head">
        <span class="panel-title">1. Low-FPR SOC ROC Curve (Alert-Fatigue Regime)</span>
        <select id="roc-mode"><option value="full_flow">Full Flow (34 vars)</option><option value="shortcut_ablated">Shortcut-Ablated (24 vars)</option></select>
      </div>
      <div id="plot-roc" style="height:360px;"></div>
    </div>
    <div class="panel">
      <div class="panel-head">
        <span class="panel-title">2. Zero-Day LOAO Detection by Unseen Attack Family</span>
        <select id="loao-budget">
          <option value="recall_at_010">Recall @ FPR &lt;= 1.0%</option>
          <option value="recall_at_005">Recall @ FPR &lt;= 0.5%</option>
          <option value="recall_at_001">Recall @ FPR &lt;= 0.1%</option>
        </select>
      </div>
      <div id="plot-loao" style="height:360px;"></div>
    </div>
    <div class="panel">
      <div class="panel-head">
        <span class="panel-title">3. TreeSHAP Behavioral Telemetry Attribution</span>
        <select id="shap-mode"><option value="full">Full Flow (34 vars)</option><option value="ablated">Shortcut-Ablated (24 vars)</option></select>
      </div>
      <div id="plot-shap" style="height:360px;"></div>
    </div>
    <div class="panel">
      <div class="panel-head"><span class="panel-title">4. OOF Calibration Reliability (Official Holdout)</span></div>
      <div id="plot-cal" style="height:360px;"></div>
    </div>
  </section>

  <section class="panel" style="margin-bottom:14px;">
    <div class="panel-head"><span class="panel-title">Official Holdout Benchmark Matrix (82,332 Flows)</span></div>
    <div class="table-wrap">
      <table><thead><tr><th>Mode</th><th>Model</th><th>Calibration</th><th class="num">ROC-AUC</th><th class="num">PR-AUC</th><th class="num">F1 Opt</th><th class="num">Brier</th><th class="num">ECE-10</th><th class="num">Rec@0.1%</th><th class="num">Rec@1.0%</th><th class="num">us/flow</th></tr></thead><tbody id="holdout-table-body"></tbody></table>
    </div>
  </section>

  <section class="panel">
    <div class="panel-head"><span class="panel-title">Zero-Day Leave-One-Attack-Family-Out Matrix (7 Unseen Families)</span></div>
    <div class="table-wrap">
      <table><thead><tr><th>Unseen Family</th><th class="num">Holdout Flows</th><th class="num">Full AUC</th><th class="num">Ablated AUC</th><th class="num">IsoForest AUC</th><th class="num">Full @0.1%</th><th class="num">Abl @0.1%</th><th class="num">Full @1.0%</th><th class="num">Abl @1.0%</th><th class="num">Iso @1.0%</th></tr></thead><tbody id="loao-table-body"></tbody></table>
    </div>
  </section>
</div>
<script>{_DASHBOARD_JS.replace("__NIDSZ_JSON__", json_blob)}</script>
</body>
</html>"""
    dashboard_path.write_text(html, encoding="utf-8")
    return dashboard_path
