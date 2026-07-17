# Adaptive Embodied AI

Most gesture interfaces assume everyone moves the same way. They don't.

This project explores the opposite: a system that learns *your* movement patterns and adapts to you, rather than forcing you to conform to a generic model.

## What if personalisation happened at the system level instead of asking users to adapt?

For people with different motor abilities, different movement habits, or just different bodies, a one-size-fits-all gesture interface isn't helpful—it's a wall. If the system learned how *you* naturally control space, interaction becomes accessible by design, not as an afterthought.

There's something deeper here too: your movement signature is information. Individual differences should be features, not bugs.

## What I'm Building

A real-time system that:
- Captures movement via webcam (no special hardware needed)
- Extracts pose features using MediaPipe
- Learns your personal movement patterns
- Compares personalised vs. generic models to measure the adaptation benefit

**Current phase:** Solid movement capture and exploratory analysis. The personalisation layer comes next—that's where the real learning happens.

## How It Works

Here's the signal flow:
```
Webcam → MediaPipe pose landmarks → Feature extraction 
→ Movement dataset → Analysis & adaptive models
```

I chose pose landmarks over raw video intentionally: they're stable, they reduce noise, and they skip straight to what matters—how your body moves through space. Torso movement tells me more about navigation intent than finger flutter ever will.

**Why this approach:**
- Accessible: runs on any webcam, no depth sensors required
- Focused: extracts just the features that matter for understanding *your* style
- Testable: I can compare what a model learns about you vs. a generic user

## What This Taught Me

Building this forced me to think carefully about **what data actually reveals about movement**. It's not just coordinates—it's consistency, compensation patterns, individual quirks. Most accessible design fails because it's built on assumptions about "normal" movement, not on actual human variation.

The other realisation: capturing variation is hard. You need enough data to find patterns, but also enough structure to extract meaning. That tension shaped everything.

## Try It

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .

python experiments/record_movement.py
```

Press `ESC` when done. Data is saved to `data/movement/`.

## Code

```
src/adaptive_embodied_ai/
├── acquisition/       — Pose capture & webcam handling
├── representation/    — Feature extraction & normalization
└── utils/

experiments/          — Movement recording scripts
notebooks/           — Exploratory analysis & visualization
data/movement/       — Your recorded sessions
```

**Stack:** Python · OpenCV · MediaPipe · NumPy, Pandas · PyTorch (coming next)
