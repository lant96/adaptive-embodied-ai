# Adaptive Embodied AI

A research prototype investigating whether webcam-based pose tracking can be personalised to how a specific person moves, rather than relying on a generic movement model.

> **Can a generic movement-recognition model be improved by learning from a small amount of an individual's own movement data — and how much data does that actually take?**

---

## What it is

Gesture-based interfaces are usually built on the assumption that people move in similar ways. In practice, movement style varies a lot between individuals — body proportions, habits, motor style. This project builds a small, transparent, end-to-end pipeline — webcam → MediaPipe pose estimation → hand-engineered movement features → interpretable classifiers — and uses it to actually test, empirically, whether personalising to an individual helps, and by how much.

Built with Python, OpenCV, MediaPipe, scikit-learn

---

## How it works

```
Webcam → MediaPipe PoseLandmarker → Pose landmarks (2D + world-space)
  → Movement feature extraction (shoulder-width normalised)
  → Trial-level aggregation (mean, std)
  → StandardScaler → Logistic Regression → Movement class
```

Note: depth-aware features are computed from MediaPipe's `pose_world_landmarks` (metric 3D, hip-centred), not the image-space `z` channel, which is documented as a rough estimate and was found empirically unreliable in an earlier version of this pipeline — see Key Findings.

---

## Key Findings

A few intentional design decisions and findings worth noting:

- **Shoulder-width normalisation matters.** Raw image-space coordinates confound "how far someone sits from the camera" with "how they move." All spatial features are normalised by shoulder width to reduce this.
- **A simpler feature set outperformed a more complex one.** Adding min/max statistics on top of mean/std slightly *hurt* generalisation with this few training trials — more features isn't automatically better on a small dataset.
- **Personalisation helps, but not uniformly.** Adding a handful of calibration trials from a target participant sharply improved one participant's accuracy (a step-function jump with just one trial per class, then a plateau) and did nothing measurable for the other — personalisation benefit appears to depend on how well an individual's baseline movement already resembles the generic population.
- **Depth-aware features didn't add value here**, once correctly computed from world landmarks. Shoulder width already implicitly encodes depth through perspective scale, which may explain why explicit depth features added nothing on top.
- All results are treated as **exploratory findings from a two-participant pilot**, evaluated with leave-one-participant-out and fixed cross-session testing — not as claims that generalise to a broader population.

---

## Architecture

```
src/adaptive_embodied_ai/
├── acquisition/          — camera capture, MediaPipe pose tracking, CSV recording
├── representation/       — movement feature extraction + normalisation
└── utils/                — paths and shared config

experiments/
├── collect_dataset.py    — cued trial-based recording protocol
├── validate_dataset.py   — schema/balance/timestamp checks before analysis
└── 01–05_*.ipynb         — exploratory analysis → baseline → personalisation
                             → calibration size → depth ablation, in that order

tests/                    — unit tests for feature extraction and normalisation
data/movement/            — recorded pose feature sessions (2 participants, 2 sessions each)
```

Read the notebooks in numeric order — each answers one specific question, building on the last.

---

## Stack

Python · OpenCV · MediaPipe (Pose Landmarker) · NumPy · Pandas · scikit-learn

---

## Running locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

Record a new session:
```bash
python experiments/collect_dataset.py --participant P03 --session S01
```

Run the tests:
```bash
pytest
```

Then open any notebook in `experiments/` to reproduce the analysis.

---

## Limitations & next steps

This is a small research prototype, not a validated system:

- Two participants, 100 recorded trials, five movement classes — findings are exploratory, not generalisable
- Trial-level aggregation (mean/std) discards temporal structure within a movement
- Personalisation was tested with static calibration sets, not online/continuous adaptation
- No Random Forest or non-linear baseline has been tested yet
- An earlier depth-feature implementation was found to be numerically unreliable and was corrected; see `05_depth_ablation.ipynb` for the full story

---

## Author

Athanasia Lantouri — MSc Data Science & Machine Learning
ath.lantouri@gmail.com