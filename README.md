# Adaptive Embodied AI
Personalized embodied navigation through human movement.

**Research Question:** Can an AI system learn how an individual controls an interaction space, adapting instead of forcing users to conform?

---

## The Idea

Most gesture interfaces assume everyone moves the same way. They don't.

This project explores the opposite: a system that learns an individual's movement characteristics and adapts in real-time.

**Input:** Webcam  
**Task:** Navigate a virtual space through natural body movement  
**Comparison:** Generic model vs. personalized model

---

## Current Implementation

**Phase 1: Movement Capture** ✅
- OpenCV camera acquisition
- MediaPipe pose tracking
- Feature extraction (normalized movement, torso position, head offset, etc.)
- Data recording

**Phase 2: Analysis & Baseline** 🔄
- Exploratory analysis of movement patterns
- Baseline model development

**Phase 3: Personalization** 📋
- User-specific adaptation
- Generic vs. personalized comparison

---

## Getting Started

```bash
# Setup
python -m venv .venv
.venv\Scripts\activate
pip install -e .

# Record movement
python experiments/record_movement.py
```

Press `ESC` to stop. Data saves to `data/movement/`.

---

## Structure

```
src/adaptive_embodied_ai/
├── acquisition/          # Pose tracking & camera
├── representation/       # Feature extraction
└── utils/

experiments/             # Data collection & training scripts
notebooks/              # Analysis & visualization
data/movement/          # Recorded sessions
```

---

## Tech Stack
- Python, OpenCV, MediaPipe
- NumPy, Pandas (analysis)
- PyTorch/scikit-learn (models, coming soon)

---

## Why This Matters
- **Accessibility:** Systems can adapt to different motor abilities
- **Personalization:** Individual differences become features, not bugs
- **Reproducibility:** Standard webcam, no special hardware

---

## Author
Athanasia Lantouri | [@lant96](https://github.com/lant96)