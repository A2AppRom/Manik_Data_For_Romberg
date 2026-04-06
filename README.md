# Romberger

A fun, educational tool that makes use of the Romberg balance test using your phone's accelerometer data.

---

## What is it?

Romberger is a digital implementation of the [Romberg test](https://en.wikipedia.org/wiki/Romberg%27s_test), a standard neurological balance assessment. Upload a CSV from your phone's sensor app, and Romberger will tell you whether your balance looks normal or "off."

This is not a diagnostic tool. It's built to make balance science accessible and help users learn about how the body maintains posture through impairment.

---

## How does it work?

The user stands still for two timed trials:

1. **Eyes open** — baseline sway recording
2. **Eyes closed** — sway recording without visual feedback

The app compares sway between the two conditions. Healthy individuals sway slightly more with eyes closed, but an excessive increase is the key Romberg signal. Romberger computes this difference and returns a simple, readable result.

---

## Tech stack

- **Python** + **Jupyter** — feature extraction and model training
- **HTML / CSS / JS** — frontend, runs entirely in the browser (no server needed)

*More to be added.*

---

## Getting started

*Setup instructions coming soon.*

---

## License

Educational use only. Not intended for clinical or medical applications.



