# WAFER-GUARD Audit — Fab-7 Excursion Detector

**Recommendation: NO-GO.** The reported F1 ≥ 0.90 comes from leakage, not from the model. After the repairs below, the model is close to chance on future wafers at natural prevalence.

All numbers below come from `wafer_guard_fixed.ipynb` (seed 1337, CPU, deterministic; two runs gave identical metrics).

---

## 1. The three silent bugs

### Bug 1 — Data leakage (severity: critical)
**Where:** original §2, preprocessing and split.
**Mechanism:**
- `SimpleImputer` and `StandardScaler` were fit on **all 1,567 rows**, including future and test lots.
- **SMOTE ran before the split.** Synthetic defects built from training defects ended up in the test set: **274 of the 293** test "defects" are synthetic.
- The test set was **50 % defective**, against a natural rate of 6.6 %.
- `train_test_split` shuffled lots from 2008-07-19 to 2008-10-17. The model was trained on the future and tested on the past.

**Evidence:** a plain logistic regression scores **F1 = 0.932** under the original protocol. Under a causal protocol the same model gets **PR-AUC 0.090**. The release gate was measuring the leak.

**Fix:**
- Rows are sorted by `timestamp` and split by time: train 60 % (Jul 19 – Sep 20), val 20 % (Sep 20 – Oct 2), test 20 % (Oct 2 – Oct 17).
- Column filtering, median imputation and scaling are fit on train rows only. `transform` is row-wise.
- SMOTE is removed. Imbalance is handled with BCE `pos_weight = 11.37`, computed from train only.
- Epoch selection and the decision threshold use validation only. Test is scored once.

### Bug 2 — Corrupted loss gradients via broadcasting (severity: critical)
**Where:** original §3 and §5. The loader `unsqueeze(1)` makes `batch_x` shape `[B, 1, F]`, while the model's `flatten` returns `reconstructed` as `[B, F]`.
**Mechanism:** `(reconstructed - batch_x)` silently broadcasts to **`[64, 64, 590]`**, so each reconstruction is compared against *every* wafer in the batch. The minimiser of that loss is the batch mean.
**Evidence:** the broken gradient has cosine **0.714** with the correct gradient and **1.000** with the "move toward the batch mean" gradient. The decoder learns to output the average wafer, and the encoder gets no per-wafer reconstruction signal.
**Fix:**
- Tensors stay flat `[B, F]` throughout.
- The loss is a masked per-element MSE with `assert recon.shape == target.shape == mask.shape`.
- There is also an assert that logits and labels are both `[B]`.

### Bug 3 — Invariant blindness: legacy probe station PS-L (severity: high)
**Where:** original §2, where all 590 sensors are treated as continuous.
**Mechanism:**
- **146 wafers (9.3 %)** have `sensor_0 == 0.0`, a PS-L sentinel. On exactly those rows `sensor_1` holds firmware codes **100 (61), 200 (35) and 300 (50)**.
- The original pipeline mean-imputed and z-scored these as measurements. That inflated σ(sensor_0) from 74 to **880** and σ(sensor_1) from 80 to **675**.
- Every PS-L wafer sat at z = −3.1, which looks like a fake excursion.
- The codes were read as ordinal magnitudes (300 counted as "more" than 100).
- PS-L wafers have a **0 % defect rate** (vs 7.3 % on current-gen), so the model could learn "station type" instead of "defect".

**Fix:**
- PS-L is detected per row with the exact `sensor_0 == 0.0` rule. Unexpected codes trigger an `assert`.
- New categorical inputs: `is_legacy` plus a one-hot `ps_l_code_{100,200,300}`. These are not scaled.
- On PS-L rows, `sensor_0` and `sensor_1` are set to *missing*. They are left out of scaler statistics and imputation, set to a neutral 0 in the input, and **masked out of the reconstruction loss**. No made-up continuous value enters training.
- The evaluator mask file was used only to confirm the rule matches it (146/146 rows). The pipeline does not read it.

---

## 2. True metrics at natural prevalence (held-out future window)

Test set: 314 wafers from 2008-10-02 to 2008-10-17, 17 defects (**5.4 % prevalence**, natural and not resampled). The threshold was frozen on validation at 0.628.

| Metric | Repaired model |
|---|---|
| **PR-AUC (average precision)** | **0.059** (no-skill baseline = 0.054, 1.09×) |
| ROC-AUC | 0.498 |
| F1 @ val threshold | 0.000 |
| Precision / Recall | 0.000 / 0.000 |
| Confusion (TN / FP / FN / TP) | 293 / 4 / 17 / 0 |

**Why these metrics:** at about 6 % prevalence, accuracy and ROC-AUC look good even for useless models. PR-AUC compared with the prevalence baseline is the main measure. F1, precision and recall are reported at a threshold chosen without test labels.

**Disclosure:**
- Hyper-parameters (hidden 256, latent 8, wd 1e-2, lr 1e-4) came from a 32-config sweep scored on **validation PR-AUC only**.
- Before that sweep, the notebook's original hyper-parameters were also run through the repaired pipeline once. That run got test PR-AUC 0.102, ROC-AUC 0.595 and F1 0.146, with a degenerate val threshold of 0.0001.
- Neither configuration was chosen using test results. Both runs are near the no-skill line.

---

## 3. Residual risks
- **Tiny validation signal:** only 11 val defects, so epoch and threshold selection is noisy. The best validation epoch was 2, meaning almost nothing learned carries forward in time.
- **Temporal drift:** the defect rate moves from 8.1 % (train) to 3.5 % (val) to 5.4 % (test).
- **Station confounding:** PS-L wafers never fail in this data. A future PS-L failure would be scored with almost no evidence.
- **Clipping and imputation:** the ±10σ clip and train-median imputation are simple choices and should be monitored in production.

## 4. Go / no-go
**NO-GO.** Do not deploy on Friday.

Before this model is reconsidered:
1. Collect more labelled excursions.
2. Use rolling-origin (walk-forward) validation.
3. Compare against simple baselines such as logistic regression or gradient boosting with feature selection.
4. Set a release gate on time-split **PR-AUC lift over prevalence**, not on F1 from a balanced test set.
