# SkyClass

Classifying Sloan Digital Sky Survey objects as **GALAXY**, **STAR** or **QSO**
(quasar) from 100,000 spectra in the
[SDSS DR17 stellar-classification dataset](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17).

Redshift alone solves almost all of this task, so reporting one accuracy number
would be misleading. The project therefore runs **two experiments**:

* **Photometric** — only what an imaging survey sees: sky position, five band
  magnitudes, four colours.
* **Spectroscopic** — the same, plus redshift.

The gap between them is the actual result.

## Results (stratified 80/20 hold-out, seed 0)

| regime | model | accuracy | macro F1 | balanced acc | log loss |
|---|---|---|---|---|---|
| photometric | **hist gradient boosting** | **0.8920** | **0.8598** | **0.8522** | 0.2992 |
| photometric | random forest | 0.8901 | 0.8572 | 0.8489 | 0.3318 |
| photometric | logistic regression | 0.7643 | 0.6823 | 0.6819 | 0.6478 |
| photometric | majority baseline | 0.5945 | 0.2485 | 0.3333 | 14.62 |
| + redshift | **random forest** | **0.9759** | **0.9720** | **0.9688** | 0.0893 |
| + redshift | hist gradient boosting | 0.9758 | 0.9718 | 0.9679 | 0.0755 |
| + redshift | logistic regression | 0.9560 | 0.9492 | 0.9467 | 0.1484 |

A **single-feature** logistic regression on redshift alone reaches **0.9250**
macro F1 — it already beats every photometric model while ignoring the images
entirely. Stars sit at redshift ~ 0, extragalactic objects do not, and that
threshold does most of the work. This is the honest headline of the project.

Removing redshift costs about **11 points of macro F1** (0.972 → 0.860). What
remains is a real astronomy problem: colours separate stars from galaxies well,
but **QSO vs GALAXY is hard** (photometric QSO F1 0.82) because faint quasars and
blue galaxies overlap in colour space.

Per-class photometric performance (best model):

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| GALAXY | 0.924 | 0.954 | 0.939 | 11,889 |
| QSO | 0.825 | 0.809 | 0.817 | 3,792 |
| STAR | 0.857 | 0.793 | 0.824 | 4,319 |

With redshift, STAR becomes nearly perfect (F1 0.994) and GALAXY rises to 0.980;
QSO (0.942) stays the hardest class.

The majority baseline's log loss is enormous (14.6) because a constant predictor
assigns **zero** probability to two classes. That is correct arithmetic, not a
bug — it is why log loss is reported alongside F1.

## How it works

1. **Clean** — drop the one row with unobserved bands. SDSS writes `-9999` for
   "not observed"; those magnitudes must never be treated as real objects.
2. **Features** — drop identifiers (`obj_ID`, `plate`, `MJD`, `fiber_ID`, ...)
   because they describe the observation, not the physics. Encode right
   ascension as sin/cos so the 0°/360° wrap does not create a false distance,
   and add four adjacent-band colours (`u-g`, `g-r`, `r-i`, `i-z`), which are
   extinction-robust and carry most of the physical signal.
3. **Split** — stratified 80/20 so rare QSOs and stars appear in both halves.
   All preprocessing that learns from data (scaling) lives inside a pipeline
   fitted on train only.
4. **Evaluate** — majority baseline, logistic regression, random forest and
   histogram gradient boosting, scored on accuracy, macro F1, balanced accuracy
   and log loss.
5. **Report** — confusion matrices, colour profiles, redshift-by-class and a
   self-contained `artifacts/report.html`.

## Usage

The repo's Python 3.12 virtualenv already has the dependencies (pandas, numpy,
scikit-learn, matplotlib, kagglehub); no new packages are needed.

```bash
cd "all machine learning projects/skyclass"
../../.venv/bin/python run_all.py                       # full study
../../.venv/bin/python -m skyclass.cli run --test-size 0.3 --seed 7
../../.venv/bin/python download_data.py                 # fetch dataset only
```

The dataset is fetched by `kagglehub` on first use and cached in
`~/.cache/kagglehub/`, never committed here.

## Outputs

`artifacts/` contains `report.html`, the four figures, `metrics_photometric.csv`,
`metrics_spectroscopic.csv` and the per-class tables.

## Limitations

* Single stratified split, not cross-validation; metrics will move by a few
  tenths of a point with a different seed.
* SDSS is not a uniform sky sample — it targets galaxies and quasars
  preferentially, so these accuracies do not transfer to an all-sky imaging
  survey.
* Hyperparameters are fixed by hand, not tuned; the point is comparing regimes,
  not squeezing the last point of F1.
* This is a teaching pipeline, not a classification service.
