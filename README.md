# Closed-Loop Cardiac & Metabolic Monitoring System

**Author:** Mehran Khosravi  
**Background:** Control & Instrumentation Engineering

## Overview
This project implements a closed-loop biosignal processing and automated drug delivery control pipeline in Python. It combines real-time electrocardiogram (ECG) feature extraction and metabolic biomarker tracking with a machine learning classifier and a proportional-integral (PI) fluidic valve controller.

The system is designed around two primary objectives:
1. **Acute Cardiac Intervention:** Detecting high-risk cardiac anomalies from heart rate variability (HRV) and blood oxygen saturation ($SpO_2$) telemetry, and automatically calculating micro-valve actuation parameters to release emergency medication.
2. **Metabolic & Dietary Regulation:** Evaluating blood glucose, lipid profiles (triglycerides), and sleep duration to provide personalized nutritional adjustments for patients and individuals managing metabolic health.

## System Architecture

* **`BiosignalProcessor`**: Applies a 4th-order Butterworth bandpass filter ($0.5\text{--}40\text{ Hz}$) to remove baseline wander and high-frequency noise from raw ECG signals, detects R-peaks, and computes time-domain HRV metrics (`HR`, `SDNN`, `RMSSD`).
* **`RandomForestClassifier`**: Evaluates multi-modal physiological vectors (`HR`, `SDNN`, `RMSSD`, `SpO2`, `Glucose`, `Triglycerides`) to estimate the probability of an acute cardiac event.
* **`ProportionalValveActuator`**: Translates the predicted cardiac risk probability and $SpO_2$ deficit into a proportional-integral control signal, regulating valve opening percentage and drug infusion flow rate ($\mu\text{L/min}$).
* **`MetabolicDietAnalyzer`**: Rule-based metabolic profiler that outputs dietary and macronutrient adjustments based on glycemic, lipid, and sleep metrics.

## Requirements

* Python 3.9+
* `numpy`
* `scipy`
* `scikit-learn`

Install dependencies using pip:

```bash
pip install numpy scipy scikit-learn
```

## Usage

Run the main simulation and control loop script:

```bash
python closed_loop_monitor.py
```

The script trains the classifier on multi-modal telemetry data, processes a simulated noisy ECG waveform, and prints the resulting valve actuation state and dietary profile to the console.