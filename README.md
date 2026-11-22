# GAD: guarded adaptive anomaly detection

GAD is an anomaly detector for industrial control systems. It keeps learning
from the plant's data, but every update is checked against a fixed reference
taken at commissioning. Poisoning can therefore move it only a limited amount,
even if nobody notices the poisoning.

This repository holds the code of the paper *GAD: Bounded Adaptation for
Anomaly Detection in Industrial Control Systems* (IFIP SEC 2027, submitted):
the detector, the scripts that build the poisoning datasets from SWaT, and the
scripts that produce the results.

## What is where

GAD processes the plant's readings one at a time, as they arrive. Each file in
`gad/` is named after the layer it belongs to.

| File | What it does |
|---|---|
| `gad/settings.py` | **every number we chose**, in plain units (see below) |
| `gad/commission.py` | fits GAD once on clean data |
| `gad/detector.py` | passes each reading through the layers |
| `gad/l0_quality.py` | L0: flags missing, out-of-range and stale readings |
| `gad/l1a_*.py` | L1a: predicts each sensor from the other sensors |
| `gad/l1b_physics.py` | L1b: predicts tank levels from mass balance |
| `gad/l2_*.py` | L2: turns the residuals into z-scores, per operating mode |
| `gad/l3_cusum.py` | L3: adds up small but lasting shifts |
| `gad/l4_*.py` | L4: thresholds and the alarm decision |
| `gad/l5_*.py` | L5: the four rules that guard every update, and the golden set |
| `gad/l6_evidence.py` | L6: one record per alarm |
| `gad/fit.py` | fits all layers; `gad/swat.py` holds everything specific to SWaT |
| `baselines/` | AdaptivePCA, MemStream and RLS, the detectors GAD is compared with |
| `evaluation/` | the metrics: attacks detected, false alarms, F1 |
| `build_datasets.py` | builds the 100 poisoning datasets from SWaT |
| `run_experiments.py` | runs all detectors on all datasets and prints the results table |

## Settings

All values we picked by hand are in `gad/settings.py` and can be changed there.
The main ones:

| Setting | Value in the paper |
|---|---|
| Update period | every 6 hours |
| Limit c on how far GAD may move from commissioning | 2 |
| Wait after an anomaly before learning again | 15 minutes |
| Longest time the physics branch ignores an unexplained reading | 10 minutes |
| False-alarm budget | 1 per day, 1 % of time in alarm |
| Half-life of the learned means and standard deviations | 12 hours |
| Clipping of each residual | 3 standard deviations |
| Alarm rule | 5 seconds to raise, 30 to clear |
| Golden set | 8 excerpts of 30 minutes |

## Data

SWaT is not included, because it may not be redistributed. It can be requested
from iTrust, Singapore University of Technology and Design
(https://itrust.sutd.edu.sg/). The paper uses the normal-operation recording of
December 2015.

## Running it

```bash
pip install -r requirements.txt
python build_datasets.py --convert --raw SWaT_Dataset_Normal_v1.xlsx
python run_experiments.py
```

The first command builds 100 datasets: four sensors, each without poisoning
and at four poisoning rates, each with five random orders of 21 attacks. The
second runs GAD, GAD without its guard, and the three baselines on all of them,
and prints the median results.

## Licence

MIT, see `LICENSE`. `baselines/memstream.py` contains a port of the MemStream
model (https://github.com/Stream-AD/MemStream), which is licensed under the
Apache License 2.0, see `baselines/LICENSE-MemStream`.
