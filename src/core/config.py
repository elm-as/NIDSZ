"""Central immutable configuration and paths for the NIDSZ pipeline."""

from dataclasses import dataclass
from pathlib import Path

TARGET_COLUMN = "label"
ATTACK_FAMILY_COLUMN = "attack_cat"
NORMAL_FAMILY_LABEL = "Normal"

# Classic UNSW-NB15 hop-count shortcuts that leak synthetic lab routing topology.
TTL_SHORTCUT_COLUMNS: tuple[str, ...] = ("sttl", "dttl", "ct_state_ttl")

# Full testbed/session-table shortcut set ablated to isolate pure behavioral telemetry.
TESTBED_SHORTCUT_COLUMNS: tuple[str, ...] = (
    "sttl",
    "dttl",
    "ct_state_ttl",
    "state",
    "swin",
    "dwin",
    "stcpb",
    "dtcpb",
    "ct_src_dport_ltm",
    "ct_dst_sport_ltm",
    "ct_flw_http_mthd",
    "ct_ftp_cmd",
    "is_sm_ips_ports",
)

CATEGORICAL_FLOW_COLUMNS: tuple[str, ...] = ("proto", "service", "state")

SOC_FPR_BUDGETS: tuple[float, ...] = (0.001, 0.005, 0.01)

LOAO_ATTACK_FAMILIES: tuple[str, ...] = (
    "Worms",
    "Shellcode",
    "Backdoor",
    "DoS",
    "Reconnaissance",
    "Exploits",
    "Fuzzers",
)

SUPERVISED_MODEL_NAMES: tuple[str, ...] = (
    "lightgbm",
    "hist_gradient_boosting",
    "logistic_regression",
)


@dataclass(frozen=True)
class ProjectPaths:
    """Resolved filesystem directories and artifact destinations for NIDSZ."""

    root: Path
    raw_dir: Path
    processed_dir: Path
    artifacts_dir: Path
    images_dir: Path
    train_parquet: Path
    test_parquet: Path
    results_json: Path
    dashboard_html: Path


@dataclass(frozen=True)
class NidszConfig:
    """Immutable scientific configuration governing NIDSZ experiments."""

    random_seed: int = 42
    calibration_folds: int = 3
    ece_bins: int = 10
    shap_sample_size: int = 2500
    latency_batch_size: int = 10000
    latency_repeats: int = 7
    soc_fpr_budgets: tuple[float, ...] = SOC_FPR_BUDGETS
    loao_families: tuple[str, ...] = LOAO_ATTACK_FAMILIES
    supervised_models: tuple[str, ...] = SUPERVISED_MODEL_NAMES


def resolve_project_paths(project_root: Path | None = None) -> ProjectPaths:
    """Resolve and return canonical NIDSZ paths anchored at `project_root`."""
    root = (
        project_root.resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[2]
    )
    raw_dir = root / "data" / "raw"
    artifacts_dir = root / "artifacts"
    return ProjectPaths(
        root=root,
        raw_dir=raw_dir,
        processed_dir=root / "data" / "processed",
        artifacts_dir=artifacts_dir,
        images_dir=root / "images",
        train_parquet=raw_dir / "UNSW_NB15_training-set.parquet",
        test_parquet=raw_dir / "UNSW_NB15_testing-set.parquet",
        results_json=artifacts_dir / "results.json",
        dashboard_html=artifacts_dir / "dashboard.html",
    )
