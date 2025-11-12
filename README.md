# Mechanical Time-Series Needle Spike Detector

Detect short-lived "needle" spikes in industrial or laboratory sensor readings with a single Python script. This repository contains `spike_detector.py`, a configuration-driven utility that scans CSV data, finds transient spikes, and exports both tabular summaries and annotated plots.

## Key Features

- **Drop-in CSV analysis** – Point the script at a folder and it will load every file that matches the configured glob.
- **Robust signal normalization** – Column headers are normalized so that `Time(s)` and `signal_*` columns are detected even if they have extra spaces or different capitalization.
- **MAD-based spike detection** – Uses a Median Absolute Deviation (MAD) based threshold with user-defined overrides for amplitude and rate.
- **Visual reporting** – Creates side-by-side plots per signal with red `×` markers on each detected spike, plus a consolidated CSV report for easy post-processing.

## Quick Start

1. **Install dependencies**
   ```bash
   pip install pandas numpy matplotlib
   ```
   (Optional) Use a virtual environment to keep dependencies isolated:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install pandas numpy matplotlib
   ```

2. **Prepare your data**
   - Place one or more CSV files in a `data/` directory at the project root.
   - Each file must contain a time column (`Time(s)` by default) and one or more signal columns (`signal_0`, `signal_1`, `signal_2` by default).

3. **Run the detector with defaults**
   ```bash
   python spike_detector.py
   ```

4. **Adjust behaviour with CLI flags (optional)** – for example, scan a custom
   folder, analyse two specific signals, increase the spike amplitude threshold,
   and skip plotting:
   ```bash
   python spike_detector.py \
     --data-dir ./my_measurements \
     --signals pressure load_cell \
     --min-abs-jump 0.35 \
     --no-plot
   ```

5. **Inspect the results**
   - A `spike_report.csv` file summarizes every detected event (file name, signal, timestamps, amplitudes, etc.). You can change
     the report name with `--report-name`.
   - When spikes are present in a file, a plot is saved to `plots/<file>_spikes.png` highlighting the peak (`t_curr`) and return-to-baseline (`t_back`) times. Disable this step with `--no-plot`.

## Configuration Overview

`spike_detector.py` exposes every runtime option both as CLI arguments and via
the `DetectorConfig` dataclass. Override them inline when calling the script or
compose a custom configuration object if you import the module elsewhere.

| Category | CLI flag(s) | Dataclass attribute | Description |
| --- | --- | --- | --- |
| Data loading | `--data-dir`, `--glob` | `data_dir`, `glob` | Folder and filename pattern to scan for CSV files. |
| Columns | `--time-col`, `--signals` | `time_col`, `signal_cols` | Expected time column and signal column names; matching is case-insensitive and ignores spacing. |
| Spike detection | `--min-abs-jump`, `--z-k`, `--rate-threshold` | `min_abs_jump`, `z_k`, `rate_threshold` | Tune amplitude and optional rate thresholds. Larger `z_k` yields stricter detection. |
| Spike shape | `--spike-max-steps`, `--spike-max-dt-factor` | `spike_max_steps`, `spike_max_dt_factor` | Control how quickly the signal must return after peaking. |
| Baseline return | `--return-frac`, `--return-abs-tol` | `return_frac`, `return_abs_tol` | Allowable deviation from the pre-spike baseline when the signal comes back down. |
| Output | `--report-name`, `--no-plot` | `report_name`, `plot` | Control the CSV report name and whether to export plots. |

After adjusting any parameter, rerun `python spike_detector.py` with the desired
flags (or re-run your wrapper script) to apply the new configuration.

## Example Output

```
检测完成，共标记 5 个 spike，结果：/absolute/path/to/spike_report.csv
      file   signal  row_index  t_prev  t_curr  t_back  x_prev  x_peak  x_back  up_jump  down_jump  dt_up  dt_down
sample.csv signal_1         42   12.34   12.35   12.37    0.12    0.48    0.15     0.36     -0.35   0.01     0.02
```

Each plotted spike is marked with a red `×`, and two dotted vertical lines indicate the peak and return-to-baseline times to help you visually confirm the detection.

## Tips

- The detector focuses on the "needle" pattern: a sharp rise followed by a fast return to the previous baseline.
- Increase `min_abs_jump` or `rate_threshold` if high-frequency noise causes false positives; lower them if true spikes are being missed.
- If your data uses a different sampling interval, adjust `spike_max_steps` and `spike_max_dt_factor` so the downstroke window matches your expectations.
- Want to skip plotting? Add `--no-plot` when running the script or set `plot=False` on the config object.
