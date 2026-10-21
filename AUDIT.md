# WAFER-GUARD Audit — Fab-7 Excursion Detector

**Recommendation: NO-GO.** The original notebook, re-run unchanged, scores F1 0.9915 and ROC-AUC 1.000 (`original/wafer_guard_original_executed.ipynb`). That score is an artefact of leakage. Repaired and evaluated causally, the model reaches **F1 0.233**, far below the 0.90 gate. All numbers come from `wafer_guard_fixed.ipynb` (`results.json`). The notebook is deterministic: two clean runs gave identical results.

## Q1 · The three silent bugs and their fixes

| # | Bug | Where | Mechanism and evidence | Fix |
|---|---|---|---|---|
| 1 | **Data leakage** | §2: imputer → scaler → SMOTE → random split | Statistics were fit on all 1,567 lots, including future and test lots. SMOTE ran before the split, so **274 of 293** test "defects" were synthetic copies of training defects. The test set was 50 % defective, and the split shuffled 3 months of drifting data. A plain logistic regression "passes" with **F1 0.932** under this protocol. | Time-ordered windows. Each lot is scaled only with statistics from earlier lots, frozen at the start of every validation/test window. No SMOTE (`pos_weight` instead). 24 h label embargo before each evaluation window. Settings, epochs and threshold come from 5-fold walk-forward validation. Test is scored once. |
| 2 | **Corrupted loss gradients** | §3 `unsqueeze(1)` + §5 `mean((reconstructed - batch_x)**2)` | `[64, 590] − [64, 1, 590]` silently broadcasts to **`[64, 64, 590]`**. The gradient points exactly toward the batch mean (cosine **1.000**), not toward each wafer (cosine 0.716). | Flat `[B, F]` tensors, masked per-element MSE, hard asserts on recon/target/mask/logit/label shapes. |
| 3 | **Invariant blindness (PS-L)** | §1–2: all 590 columns treated as continuous | 146 lots have the sentinel `sensor_0 == 0.0`, and on exactly those lots `sensor_1` is a firmware code (100/200/300). Treating them as numbers inflated σ(sensor_0) from 74 to 880 and σ(sensor_1) from 80 to 675, put every PS-L lot at z = −3.1, and read code 300 as "3× code 100". | Per-row detection with an assert on unknown codes. `is_legacy` + one-hot code as categorical inputs. On PS-L lots both channels are set to missing, so they are excluded from every statistic, never imputed into the target, and masked out of the loss. |

## Q2 · True metrics at natural prevalence

Test window: the last 20 % of lots in time (313 lots, 2008-10-02 → 10-17), **17 defects = 0.054 prevalence**. That is the natural rate for that period; the dataset average is 6.6 %, and the rate drifts over time. The test set was not resampled. The threshold **0.003** was frozen on pooled walk-forward validation before the test was scored.

| Metric | Repaired SAE | Logistic baseline (same protocol) |
|---|---|---|
| **F1** | **0.233** (95 % bootstrap CI 0.061 – 0.391) | 0.154 |
| Precision / Recall | 0.192 / 0.294 | 0.222 / 0.118 |
| **PR-AUC** (no-skill = 0.054) | **0.171** (CI 0.084 – 0.368), 3.2× lift | 0.141 |
| ROC-AUC | 0.747 | 0.724 |
| TN / FP / FN / TP | 275 / 21 / 12 / 5 | — |

**Is the F1 real?** The notebook (§9) checks it six ways:

1. F1 recomputed by hand from the confusion matrix matches sklearn.
2. The label direction is verified (flipped scores give ROC-AUC 0.25).
3. A positive control, trained on the test lots themselves, reaches F1 1.000, so the code can score high.
4. The best F1 achievable at *any* threshold is only 0.258, so the threshold is not the problem.
5. With only 17 test defects the bootstrap interval is wide (0.061 – 0.391).
6. Five other seeds give F1 0.125 – 0.200 and PR-AUC 0.127 – 0.210.

## Residual risks

- **Weak and uncertain signal:** 17 test defects, wide CIs.
- **Calibration shift:** fold models alarm on 25 % of validation lots, the refit model on 8 % of test lots.
- **Frozen statistics:** the scaling statistics in `artifacts/` are frozen at 2008-10-02 and must be refreshed as the line drifts.
- **PS-L confounding:** PS-L lots never fail in this data, and two PS-L lots (rows 109, 1203) carry extreme values that only robust scaling and the ±10 clip contain.
- **Disclosure:** this is the third repaired version. Earlier versions scored test F1 0.146 and 0.000. Both used a single 11-defect validation window and scaling statistics fit on the whole training window, and the second picked an epoch-2, barely trained model. The method was redesigned for those methodological reasons, and every v3 setting was fixed before its single test run.
- **Reference mask:** the PS-L rule was derived from the telemetry. It agrees with the reference mask supplied with the task on all 146 lots, and the pipeline never reads that file.

## Go / no-go

**NO-GO.** The detector carries real but weak signal (3× better than chance, ROC-AUC 0.75). It would catch roughly 3 in 10 excursions with about 4 false alarms per true alarm.

Before resubmitting:
1. Collect more labelled excursions.
2. Replace the F1-on-balanced-data gate with a time-split PR-AUC gate.
3. Add drift monitoring for the scaling statistics and the PS-L share.
