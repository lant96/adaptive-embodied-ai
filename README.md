# Adaptive Embodied AI

Adaptive Embodied AI is a research project exploring personalized movement-based interaction. Rather than relying on fixed gesture models, the system learns how an individual user controls an interaction space and adapts accordingly.

**Research Question**

> Can an AI system learn how an individual controls an interaction space, adapting to the user instead of forcing the user to adapt to the system?

---

## Overview

Most gesture-based interfaces assume that people move in similar ways. In practice, movement patterns differ considerably between individuals due to physiology, habits, experience, and motor abilities.

This project investigates whether interaction systems can learn these individual characteristics and build personalized movement models that improve usability while reducing the need for predefined interaction rules.

The long-term goal is to develop adaptive embodied interfaces that continuously learn from user behaviour instead of relying on generic interaction models.

---

## Current Status

### Phase 1 — Movement Capture (Complete)

- Webcam acquisition using OpenCV
- Real-time pose estimation with MediaPipe
- Feature extraction from upper-body movement
- Movement recording pipeline

### Phase 2 — Movement Analysis (In Progress)

- Exploratory data analysis
- Baseline movement modelling
- Feature evaluation

### Phase 3 — Personalized Adaptation (Planned)

- User-specific movement models
- Generic versus personalized model comparison
- Quantitative evaluation of interaction performance

---

## Project Structure

```
src/adaptive_embodied_ai/
├── acquisition/          # Camera and pose estimation
├── representation/       # Movement feature extraction
└── utils/

experiments/              # Recording and training scripts
notebooks/                # Analysis and visualization
data/movement/            # Recorded sessions
```

---

## Tech Stack

- Python
- OpenCV
- MediaPipe
- NumPy
- Pandas
- scikit-learn
- PyTorch (planned)

---

## Getting Started

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

Record a movement session:

```bash
python experiments/record_movement.py
```

Press `ESC` to stop recording. Data will be stored in `data/movement/`.

---

## Future Work

- Train personalized movement models
- Compare personalized and generic interaction strategies
- Evaluate adaptation across multiple users
- Investigate accessibility-oriented interaction
- Explore online learning for continuous personalization

---

## Author

Athanasia Lantouri

Applied Machine Learning | Human-Centered AI | Interactive Systems

GitHub: https://github.com/lant96
