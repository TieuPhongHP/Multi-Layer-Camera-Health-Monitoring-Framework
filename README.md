# ML-CHM Reproducibility Package

This package accompanies the manuscript (ICDSAIA 2026, paper ID 252) **“From Online Status to Mission Readiness: A Dependency-Aware Multi-Layer Camera Health Monitoring Framework for Large-Scale VMS.”**

## Scope and limitation

The experiment is a **synthetic, discrete-event architecture benchmark**. It evaluates orchestration behavior—dependency-aware root-cause assignment, context gates, hysteresis, episode suppression, missing telemetry, and server escalation. It does **not** estimate pixel-level detector accuracy or field performance on operational VMS video.

## Experimental design

- 10 synthetic sites, 8 heterogeneous cameras per site, 10 days per camera (800 camera-days)
- 7 development sites and 3 site-disjoint test sites (240 camera-days)
- 112 labeled fault episodes in the test sites
- Five technical layers (L0–L4), legitimate operational nuisances, and 5% baseline missing telemetry
- Fixed random seed: `20260728`

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python camhealth_experiment.py --outdir reproduced_results
```

The output directory contains configuration, calibrated thresholds, predicted episodes, event matching files, summary metrics, and missingness robustness results.

## Key files

- `camhealth_experiment.py`: complete simulation and evaluation code
- `results/simulation_config.json`: simulation configuration
- `results/thresholds.json`: thresholds learned only on development sites
- `results/summary_metrics.csv`: baseline and ablation results
- `results/robustness_missingness.csv`: additional missing-telemetry experiment
- `results/all_true_events.csv`, `results/test_true_events.csv`: synthetic event labels
- `results/predictions/`: predicted episodes for each method
- `results/matches/`: one-to-one event matching outputs
- `figures_en/`, `figures_vi/`: manuscript figures

## Stress tests of the dependency assumption (camera-ready revision)

`sensitivity_experiment.py` re-runs the unchanged detectors and calibration on generator variants that weaken or violate the lower-to-higher propagation assumed by ML-CHM (reviewer request). It does not change the main results.

```bash
python sensitivity_experiment.py --outdir sensitivity_results
```

Variants: S0 reference; S1 weak propagation (decay 0.45); S2 no propagation; S3 lower-layer artefacts with p = 0.50; S4 concurrent independent faults with p = 0.25. Outputs: `sensitivity_metrics.csv`, `sensitivity_config.json` (pre-computed copies in `results/sensitivity/`). Table 6 of the paper reports S0, S2, S3 and S4.

## Threshold guardrails

`thresholds_from_dev` clips the clean 99.7th percentile to [0.58, 0.74] (L0–L1), [0.62, 0.78] (L2) and [0.64, 0.80] (L3–L4). In this benchmark the percentiles (0.18, 0.17, 0.54, 0.50, 0.19) fall below the floors, so every threshold equals its floor.

## Main reproduced result

On the three site-disjoint test sites, full ML-CHM obtains event F1 = 0.933, event recall = 0.991, root-cause accuracy = 0.964, and 6.25 false alarms per 100 camera-days. Relative to flat multi-signal fusion, it reduces false alarms by 80.5% and server escalations by 42.2%. These values are properties of the stated synthetic benchmark, not claims of real deployment performance.

---

# Gói tái lập ML-CHM

Đây là **mô phỏng sự kiện rời rạc bằng dữ liệu tổng hợp**, dùng để kiểm tra hành vi kiến trúc: suy luận phụ thuộc, cổng ngữ cảnh, hysteresis, gom episode, tín hiệu thiếu và lượt chuyển xử lý lên máy chủ. Kết quả không đại diện cho độ chính xác bộ dò trên ảnh/video thực tế.

Chạy lại bằng lệnh:

```bash
pip install -r requirements.txt
python camhealth_experiment.py --outdir reproduced_results
```

Kết quả chính trên ba địa điểm kiểm thử độc lập: F1 theo sự kiện 0,933; recall 0,991; độ chính xác nguyên nhân gốc 0,964; 6,25 cảnh báo sai/100 camera-ngày. So với hợp nhất phẳng, cảnh báo sai giảm 80,5% và lượt chuyển lên máy chủ giảm 42,2%. Các số liệu chỉ có ý nghĩa trong cấu hình mô phỏng đã công bố.
