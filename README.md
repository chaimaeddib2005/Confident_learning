# Confident learning under class-conditional vs instance-dependent noise

Project code + first version of the presentation.

* **Old assumption** (the paper): class-conditional noise, `p(y~ | y*; x) = p(y~ | y*)`  -> `alpha = 0`
* **Our assumption**: instance-dependent noise, flip probability grows with example hardness -> `alpha > 0`

```
q_j(x) = min(1, tau_j * (1 + alpha) * u_x ** alpha)
```

`u_x` in (0,1) is the within-class percentile of hardness (1 - p_hat(y*|x), model trained on clean labels).
Because `u_x` is uniform inside a class, the class-level noise rate stays `tau_j` for every `alpha`:
only the dependence of the noise on `x` changes, so the joint `Q_{y~,y*}` is the same across `alpha`.

## Layout

```
src/cl_core.py          confident joint, bucketed confident joint, C_confusion baseline,
                        calibration (joint estimate), derived statistics, error flagging, metrics
src/noise.py            destination matrix (sparsity), hardness percentile, class-conditional and
                        instance-dependent noise (common random numbers across alpha)
src/probs.py            out-of-sample probabilities: "logreg" (features) or "cnn" (small ResNet)
src/data.py             CIFAR-10 loader, frozen ResNet-18 feature extractor, offline `digits` smoke set
src/run_experiments.py  the experiment grid -> results/results.csv
src/make_plots.py       summary table + figures (F1 / precision / recall / RMSE vs alpha)
tests/test_cl.py        unit tests (worked example, Theorem 1 exact recovery, noise properties)
slides/                 Beamer source + compiled PDF (11 slides, ~10 min)
docs/                   the full course notes (LaTeX + PDF) on the paper's math
results_smoke/          output of a smoke run on the offline digits data (NOT CIFAR-10, not for the talk)
```

## Setup

```
pip install -r requirements.txt
python -m pytest tests -q          # 8 tests, a few seconds
```

## Run on CIFAR-10 (the paper's dataset)

Fast option (frozen ImageNet ResNet-18 features + logistic regression, 4-fold cross-validation):

```
python src/run_experiments.py --dataset cifar10 --backend logreg --data-root data \
       --noise 0.2 0.4 --alphas 0 0.5 1 --sparsity 0.4 --buckets 3 5 --seeds 3 --eval-acc
python src/make_plots.py
```

Closer to the paper (small ResNet trained from scratch inside each of the 4 folds; use a GPU):

```
python src/run_experiments.py --dataset cifar10 --backend cnn --epochs 20 --seeds 1
python src/make_plots.py
```

The first command downloads CIFAR-10 and the ResNet-18 weights (needs internet) and caches the features.
Then recompile the slides (`cd slides && pdflatex presentation.tex` twice): the results slide picks up
`results/f1_vs_alpha.pdf` automatically.

Cost: the CNN backend trains `4 folds x (#noise levels x #alphas) x #seeds` networks plus one clean-label
run for the hardness scores. Start with `--seeds 1`.

## Methods compared

| name | rule |
|---|---|
| `confusion` | guess `y* = argmax p_hat` (baseline `C_confusion`) |
| `cl` | standard confident joint, per-class thresholds `t_j` |
| `cl_bucket{B}` | our fix: one threshold per (class, entropy bucket), `B` buckets; `B=1` equals `cl` |

Metrics (ground truth is known because the noise is synthetic): detection precision / recall / F1,
RMSE between the estimated joint and the realized joint, estimated error rate, mean hardness of missed
errors (`u_missed`) and of false positives (`u_false_pos`), and test accuracy after cleaning
(`--eval-acc`, logreg backend only).

## What has and has not been verified

* Verified by tests: the worked example from the course (thresholds, confident joint, calibration,
  flagged examples, class weights), the paper's Figure-1 joint, exact recovery under ideal probabilities
  (Theorem 1), the mean-preserving property of the noise model for every `alpha`, `B=1 == standard CL`.
* Smoke-tested end to end on the offline `digits` dataset and the CNN backend on random data.
* **Not run here:** CIFAR-10 (no internet in the build environment) and the pretrained ResNet-18 download.
  The slides therefore contain a placeholder on the results slide. Do not quote the `results_smoke/`
  numbers as CIFAR-10 results.
* The effects described in the slides (recall drop under `alpha > 0`, partial recovery with bucketing)
  are hypotheses; on the small digits smoke set the recall drop appears for standard CL and bucketing
  recovers recall, but bucketing performs about the same as the plain argmax baseline there. Expect to
  report whatever CIFAR-10 shows.

## Talk timing (about 10 min)

Title 0:20 | problem + assumption 1:00 | count (thresholds, confident joint) 1:40 | calibrate/prune/train 1:20 |
theory 1:00 | our noise model 1:20 | why it should break 1:00 | bucketed thresholds 0:50 | protocol 0:40 |
results 1:00 | conclusion 0:30.
