# Closed-Loop Biosignal Control & Telemetry Monitor

A real-time physiological signal processing and closed-loop actuation pipeline developed in Python. This system processes raw electrocardiogram (ECG) data, extracts time-domain heart rate variability (HRV) metrics, assesses acute cardiac risks via supervised machine learning, and dynamically modulates a micro-valve actuator for emergency therapeutic delivery alongside metabolic telemetry monitoring.

---

## System Architecture

The pipeline consists of four interconnected core modules:

1. **Biosignal Processing Engine (`BiosignalProcessor`)**
   - Applies a 3rd-order Butterworth bandpass filter (0.5 – 40.0 Hz) to eliminate baseline wander and high-frequency EMG noise.
   - Executes dynamic threshold peak detection to identify R-peaks and compute real-time RR intervals.
   - Calculates statistical time-domain HRV markers: Heart Rate (BPM), SDNN, and RMSSD.

2. **Physiologic Risk Classifier (`DiagnosticModel`)**
   - Evaluates multi-parametric physiological telemetry using an ensemble classifier trained on normative and acute cardiovascular event profiles.
   - Computes continuous risk probability scores based on real-time biometric inputs.

3. **Closed-Loop Actuation Controller (`ClosedLoopActuator`)**
   - Employs PID-based control logic to regulate micro-valve aperture percentage.
   - Translates detected risk levels into calibrated infusion flow rates ($\mu\text{L/min}$) for autonomous drug delivery upon acute anomaly detection.

4. **Interactive Telemetry Dashboard (`MonitorGUI`)**
   - Native desktop interface built with Tkinter and Matplotlib.
   - Real-time visualization of filtered ECG waveforms with annotated R-peaks.
   - Live telemetry indicators for heart rate, HRV metrics, $SpO_2$ saturation, and glycemic indices.
   - Interactive simulation controls to toggle between stable baseline and acute cardiac events.
   - Real-time metabolic and dietary guidance logging.

---

## Repository Structure

```text
closed-loop-biosignal-control/
│
├── closed_loop_monitor.py   # Complete processing, ML classification, control loop, and GUI
├── requirements.txt         # Required Python dependencies
└── README.md                # Project documentation and architecture overview
Installation & SetupPrerequisitesPython 3.9 or higherPowerShell, Command Prompt, or Terminal1. Clone or Download RepositoryBashgit clone [https://github.com/USERNAME/closed-loop-biosignal-control.git](https://github.com/USERNAME/closed-loop-biosignal-control.git)
cd closed-loop-biosignal-control
2. Install DependenciesBashpip install -r requirements.txt
3. Run the ApplicationBashpython closed_loop_monitor.py
Interactive ControlsSet Normal Baseline: Simulates healthy resting vitals ($HR \approx 72\text{ BPM}$, normative HRV, $SpO_2 \ge 98\%$). The micro-actuator remains in STANDBY mode.Simulate Acute Cardiac Event: Simulates acute tachycardia, depressed HRV, and biomarker drops ($SpO_2 < 90\%$). The system triggers emergency closed-loop intervention, activating the micro-valve to 100% flow rate and logging clinical alerts.License & Intellectual PropertyCopyright (c) 2026 Mehran Khosravi. All rights reserved.This repository represents proof-of-concept computational modeling and is provided strictly for academic and graduate research evaluation.
