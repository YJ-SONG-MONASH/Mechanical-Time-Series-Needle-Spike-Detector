"""Spike detection utility for mechanical time-series signals.

This script scans one or more CSV files that contain time-aligned signals and
identifies "needle-like" transient spikes (sharp rises followed by an equally
fast return to baseline). It can be run as a module or invoked from the command
line – defaults can be overridden through CLI flags so the detector can be
adapted to new datasets without editing the source file.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ===== 基本配置（默认值，可通过 CLI 覆盖） =====
DEFAULT_DATA_DIR = Path("./data")      # CSV 目录
DEFAULT_GLOB = "*.csv"                 # 批量：*.csv；如需单文件检测改成具体文件名
DEFAULT_TIME_COLUMN = "Time(s)"
DEFAULT_SIGNAL_COLUMNS = ["signal_0", "signal_1", "signal_2"]

# ===== 检测阈值（可调）=====
DEFAULT_MIN_ABS_JUMP = 0.25     # 最小上升/下降跳幅（单位=信号单位）
DEFAULT_Z_K = 8.0               # MAD 鲁棒倍数（越大越严格）
DEFAULT_RATE_THRESHOLD: Optional[float] = None  # 最小上升/下降速率(单位/秒)，不用则 None

# ===== “针状突变”判定参数（可调）=====
DEFAULT_SPIKE_MAX_STEPS = 2         # 上升后最多在多少采样点内出现快速回落
DEFAULT_SPIKE_MAX_DT_FACTOR = 2.5   # 最大回落时间 = 该因子 * median(dt)
DEFAULT_RETURN_FRAC = 0.35          # 回落到基线的相对容差（相对上升幅）
DEFAULT_RETURN_ABS_TOL = 0.10       # 回落到基线的绝对容差（单位）

# ===== 绘图样式（更长；× 更小；× 用红色）=====
PLOT_FIGSIZE = (20, 7)     # 更“拉长”横向
SPIKE_MARKER_SIZE = 14     # × 标记大小（越小越小）
SPIKE_LINEWIDTH = 0.9      # × 线宽
SPIKE_COLOR = "red"        # × 颜色（你要的红色）
XLABEL_FONTSIZE = 10
TICK_LABELSIZE = 9
SAVE_DPI = 180

# ----------------- IO 与列名处理 -----------------


@dataclass(frozen=True)
class DetectorConfig:
    """Configuration bundle for running the spike detector."""

    data_dir: Path = DEFAULT_DATA_DIR
    glob: str = DEFAULT_GLOB
    time_col: str = DEFAULT_TIME_COLUMN
    signal_cols: Sequence[str] = tuple(DEFAULT_SIGNAL_COLUMNS)
    min_abs_jump: float = DEFAULT_MIN_ABS_JUMP
    z_k: float = DEFAULT_Z_K
    rate_threshold: Optional[float] = DEFAULT_RATE_THRESHOLD
    spike_max_steps: int = DEFAULT_SPIKE_MAX_STEPS
    spike_max_dt_factor: float = DEFAULT_SPIKE_MAX_DT_FACTOR
    return_frac: float = DEFAULT_RETURN_FRAC
    return_abs_tol: float = DEFAULT_RETURN_ABS_TOL
    plot: bool = True
    report_name: str = "spike_report.csv"


def robust_read_csv(fp: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(fp, sep=None, engine="python")
    except Exception:
        for sep in ["\t", ";", ","]:
            try:
                return pd.read_csv(fp, sep=sep)
            except Exception:
                pass
        raise

def normalize_columns(df: pd.DataFrame, time_col: str, signal_cols: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    cols = [c.strip() for c in df.columns]
    lower = {c.lower(): c for c in cols}
    mapping = {}
    for want in [time_col, *signal_cols]:
        key = want.lower()
        if key in lower:
            mapping[lower[key]] = want
        else:
            cands = [c for c in cols if c.replace(" ", "").lower() == key.replace(" ", "")] 
            if cands:
                mapping[cands[0]] = want
    df.rename(columns=mapping, inplace=True)
    return df

# ----------------- 核心：针状突变检测 -----------------


def detect_spikes(
    df: pd.DataFrame,
    col: str,
    time_col: str,
    *,
    min_abs_jump: float,
    z_k: float,
    rate_threshold: Optional[float],
    spike_max_steps: int,
    spike_max_dt_factor: float,
    return_frac: float,
    return_abs_tol: float,
) -> List[Dict]:
    """
    仅检测“针状突变”(up-then-down)：
      1) 上升：s[i]-s[i-1] >= up_thr（可叠加速率阈值）
      2) 回落：在 i..i+spike_max_steps 内存在 j，使 s[j+1]-s[j] <= -down_thr，
         且 (t[j+1]-t[i]) <= spike_max_dt
      3) 回到基线：|s[j+1]-s[i-1]| <= max(return_abs_tol, return_frac * 上升幅)
    """
    sub = df[[time_col, col]].copy()
    sub[time_col] = pd.to_numeric(sub[time_col], errors="coerce")
    sub[col] = pd.to_numeric(sub[col], errors="coerce")
    sub = sub.dropna(subset=[time_col, col])
    if len(sub) < 3:
        return []

    sub = sub.sort_values(time_col, kind="mergesort").reset_index(drop=True)
    t = sub[time_col].to_numpy(dtype=float)
    s = sub[col].to_numpy(dtype=float)

    diff = s[1:] - s[:-1]
    dt = t[1:] - t[:-1]

    diffs_for_scale = diff[np.isfinite(diff)]
    if diffs_for_scale.size == 0:
        sigma, med = 0.0, 0.0
    else:
        med = float(np.median(diffs_for_scale))
        mad = float(np.median(np.abs(diffs_for_scale - med)))
        sigma = 1.4826 * mad

    # 上升/下降阈值
    thr_from_mad = (med + z_k * sigma) if sigma > 0 else 0.0
    up_thr = max(min_abs_jump, thr_from_mad)
    down_thr = up_thr

    with np.errstate(divide="ignore", invalid="ignore"):
        rate = diff / dt
    rate[~np.isfinite(rate)] = np.nan

    finite_dt = dt[np.isfinite(dt) & (dt > 0)]
    med_dt = np.median(finite_dt) if finite_dt.size else 0.0
    spike_max_dt = spike_max_dt_factor * med_dt if med_dt > 0 else np.inf

    events: List[Dict] = []
    n = len(s)
    for i in range(1, n - 1):
        up = s[i] - s[i - 1]
        dt_up = t[i] - t[i - 1]
        if not (up >= up_thr):
            continue
        if (rate_threshold is not None) and (dt_up > 0) and ((up / dt_up) < rate_threshold):
            continue

        found = False
        last_j = min(i + spike_max_steps, n - 2)  # j 对应 diff[j] = s[j+1]-s[j]
        for j in range(i, last_j + 1):
            down = s[j + 1] - s[j]
            dt_down = t[j + 1] - t[j]
            if down <= -down_thr:
                if rate_threshold is not None and dt_down > 0 and ((-down / dt_down) < rate_threshold):
                    continue
                if (t[j + 1] - t[i]) <= spike_max_dt:
                    baseline = s[i - 1]
                    if abs(s[j + 1] - baseline) <= max(return_abs_tol, return_frac * up):
                        events.append({
                            "row_index": int(i),
                            "t_prev": float(t[i - 1]),
                            "t_curr": float(t[i]),
                            "t_back": float(t[j + 1]),
                            "x_prev": float(s[i - 1]),
                            "x_peak": float(s[i]),
                            "x_back": float(s[j + 1]),
                            "up_jump": float(up),
                            "down_jump": float(down),
                            "dt_up": float(dt_up),
                            "dt_down": float(dt_down),
                        })
                        found = True
                        break
        if not found:
            continue

    return events

# ----------------- 画图（× 为红色，小号；画布更宽） -----------------


def plot_file_with_spikes(
    df: pd.DataFrame,
    events_for_file: List[Dict],
    file_name: str,
    *,
    time_col: str,
    signal_cols: Sequence[str],
) -> None:
    if not events_for_file:
        return

    df = df.copy()
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    for c in signal_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)

    ev_by_sig: Dict[str, List[Dict]] = {c: [] for c in signal_cols}
    for e in events_for_file:
        ev_by_sig[e["signal"]].append(e)

    fig, axes = plt.subplots(len(signal_cols), 1, figsize=PLOT_FIGSIZE, sharex=True)
    if len(signal_cols) == 1:
        axes = [axes]

    for i, col in enumerate(signal_cols):
        if col not in df.columns:
            axes[i].set_title(f"{col} (missing)")
            continue

        axes[i].plot(df[time_col].to_numpy(), df[col].to_numpy(), linewidth=1.1)
        axes[i].grid(True, linestyle="--", alpha=0.3)
        axes[i].set_ylabel(col)
        axes[i].tick_params(axis="both", labelsize=TICK_LABELSIZE)

        evs = ev_by_sig.get(col, [])
        if evs:
            t_marks = [e["t_curr"] for e in evs]
            y_marks = [e["x_peak"] for e in evs]
            axes[i].scatter(
                t_marks, y_marks,
                marker="x",
                s=SPIKE_MARKER_SIZE,
                linewidths=SPIKE_LINEWIDTH,
                color=SPIKE_COLOR,       # 关键：× 用红色
                label="spike"
            )
            for e in evs:
                axes[i].axvline(e["t_curr"], linestyle=":", alpha=0.25)
                axes[i].axvline(e["t_back"], linestyle=":", alpha=0.25)
            axes[i].legend(loc="best", fontsize=9)

    axes[-1].set_xlabel(time_col, fontsize=XLABEL_FONTSIZE)
    fig.suptitle(f"{file_name} — detected spikes")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    outdir = Path("plots"); outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"{Path(file_name).stem}_spikes.png"
    fig.savefig(outpath, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {outpath.resolve()}")

def run_detector(config: DetectorConfig) -> None:
    """Execute the spike detector end-to-end using the supplied configuration."""

    all_reports: List[Dict] = []
    files = sorted(config.data_dir.glob(config.glob))
    if not files:
        print(f"[WARN] 没找到CSV：{config.data_dir / config.glob}")

    per_file_events: Dict[Path, List[Dict]] = {}

    for fp in files:
        try:
            df = robust_read_csv(fp)
        except Exception as exc:
            print(f"[SKIP] {fp.name} 读取失败：{exc}")
            continue

        df = normalize_columns(df, config.time_col, config.signal_cols)
        need = [config.time_col, *config.signal_cols]
        missing = [c for c in need if c not in df.columns]
        if missing:
            print(f"[SKIP] {fp.name} 缺列 {missing}")
            continue

        file_events: List[Dict] = []
        for col in config.signal_cols:
            events = detect_spikes(
                df,
                col,
                config.time_col,
                min_abs_jump=config.min_abs_jump,
                z_k=config.z_k,
                rate_threshold=config.rate_threshold,
                spike_max_steps=config.spike_max_steps,
                spike_max_dt_factor=config.spike_max_dt_factor,
                return_frac=config.return_frac,
                return_abs_tol=config.return_abs_tol,
            )
            for event in events:
                all_reports.append({"file": fp.name, "signal": col, **event})
                file_events.append({"signal": col, **event})
        per_file_events[fp] = file_events

    report_df = pd.DataFrame(all_reports)
    if not report_df.empty:
        report_df = report_df.sort_values(["file", "signal", "t_curr", "row_index"])

    out_path = Path(config.report_name)
    report_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"检测完成，共标记 {len(report_df)} 个 spike，结果：{out_path.resolve()}")
    if not report_df.empty:
        print(report_df.head(20).to_string(index=False))

    if not config.plot:
        return

    # 只为“有 spike”的文件出图
    for fp, evs in per_file_events.items():
        if not evs:
            continue
        try:
            df = robust_read_csv(fp)
            df = normalize_columns(df, config.time_col, config.signal_cols)
            plot_file_with_spikes(
                df,
                evs,
                fp.name,
                time_col=config.time_col,
                signal_cols=config.signal_cols,
            )
        except Exception as exc:
            print(f"[PLOT-SKIP] {fp.name} 出图失败：{exc}")


def _float_or_none(value: str) -> Optional[float]:
    """Parse floats that may also be the literal string "none" (case-insensitive)."""

    if value.lower() in {"none", "null", "na", "nan"}:
        return None
    return float(value)


def parse_args(argv: Optional[Sequence[str]] = None) -> DetectorConfig:
    """Parse command-line arguments into a :class:`DetectorConfig`."""

    parser = argparse.ArgumentParser(
        description="Detect needle-like spikes in mechanical time-series data."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="CSV 文件目录 (默认: %(default)s)",
    )
    parser.add_argument(
        "--glob",
        default=DEFAULT_GLOB,
        help="匹配 CSV 的 glob 模式 (默认: %(default)s)",
    )
    parser.add_argument(
        "--time-col",
        default=DEFAULT_TIME_COLUMN,
        help="时间列名 (默认: %(default)s)",
    )
    parser.add_argument(
        "--signals",
        nargs="+",
        default=list(DEFAULT_SIGNAL_COLUMNS),
        help="要检测的信号列 (默认: %(default)s)",
    )
    parser.add_argument(
        "--min-abs-jump",
        type=float,
        default=DEFAULT_MIN_ABS_JUMP,
        help="最小上升/下降跳幅 (默认: %(default)s)",
    )
    parser.add_argument(
        "--z-k",
        type=float,
        default=DEFAULT_Z_K,
        help="MAD 鲁棒倍数 (默认: %(default)s)",
    )
    parser.add_argument(
        "--rate-threshold",
        type=_float_or_none,
        default=DEFAULT_RATE_THRESHOLD,
        help="最小上升/下降速率，none 表示关闭 (默认: %(default)s)",
    )
    parser.add_argument(
        "--spike-max-steps",
        type=int,
        default=DEFAULT_SPIKE_MAX_STEPS,
        help="允许的最大快速回落步数 (默认: %(default)s)",
    )
    parser.add_argument(
        "--spike-max-dt-factor",
        type=float,
        default=DEFAULT_SPIKE_MAX_DT_FACTOR,
        help="最大回落时间因子 (默认: %(default)s)",
    )
    parser.add_argument(
        "--return-frac",
        type=float,
        default=DEFAULT_RETURN_FRAC,
        help="回落到基线的相对容差 (默认: %(default)s)",
    )
    parser.add_argument(
        "--return-abs-tol",
        type=float,
        default=DEFAULT_RETURN_ABS_TOL,
        help="回落到基线的绝对容差 (默认: %(default)s)",
    )
    parser.add_argument(
        "--report-name",
        default="spike_report.csv",
        help="导出报告的文件名 (默认: %(default)s)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="只生成 CSV 报告，不生成图像",
    )

    args = parser.parse_args(argv)

    return DetectorConfig(
        data_dir=args.data_dir,
        glob=args.glob,
        time_col=args.time_col,
        signal_cols=tuple(args.signals),
        min_abs_jump=args.min_abs_jump,
        z_k=args.z_k,
        rate_threshold=args.rate_threshold,
        spike_max_steps=args.spike_max_steps,
        spike_max_dt_factor=args.spike_max_dt_factor,
        return_frac=args.return_frac,
        return_abs_tol=args.return_abs_tol,
        report_name=args.report_name,
        plot=not args.no_plot,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    config = parse_args(argv)
    run_detector(config)


if __name__ == "__main__":
    main()
