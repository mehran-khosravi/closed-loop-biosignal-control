import sys
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.ensemble import RandomForestClassifier

import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class BiosignalProcessor:
    def __init__(self, sampling_rate=250):
        self.fs = sampling_rate

    def bandpass_filter(self, signal, lowcut=0.5, highcut=40.0, order=3):
        nyquist = 0.5 * self.fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, signal)

    def extract_features(self, ecg_window):
        filtered = self.bandpass_filter(ecg_window)
        peaks, _ = find_peaks(filtered, distance=self.fs * 0.4, height=np.mean(filtered) + 0.3 * np.std(filtered))
        
        if len(peaks) > 1:
            rr_intervals_ms = (np.diff(peaks) / self.fs) * 1000.0
            hr_bpm = 60000.0 / np.mean(rr_intervals_ms)
            sdnn_ms = np.std(rr_intervals_ms)
            rmssd_ms = np.sqrt(np.mean(np.diff(rr_intervals_ms) ** 2)) if len(rr_intervals_ms) > 1 else 0.0
        else:
            hr_bpm = 70.0
            sdnn_ms = 40.0
            rmssd_ms = 30.0

        return filtered, peaks, {
            'hr_bpm': hr_bpm,
            'sdnn_ms': sdnn_ms,
            'rmssd_ms': rmssd_ms
        }


class ClosedLoopActuator:
    def __init__(self, kp=1.2, ki=0.05, kd=0.1, max_flow_ul_min=50.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_flow = max_flow_ul_min
        self.integral = 0.0
        self.prev_error = 0.0

    def compute_actuation(self, risk_score):
        error = risk_score - 0.20
        if error > 0:
            self.integral += error
            derivative = error - self.prev_error
            control_signal = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
            self.prev_error = error
            valve_pct = np.clip(control_signal * 100.0, 0.0, 100.0)
        else:
            self.integral = 0.0
            self.prev_error = 0.0
            valve_pct = 0.0

        flow_rate = (valve_pct / 100.0) * self.max_flow
        status = "ACTIVE_INFUSION" if valve_pct > 10.0 else "STANDBY"
        
        return {
            'valve_open_pct': valve_pct,
            'flow_rate_ul_min': flow_rate,
            'status': status
        }


class DiagnosticModel:
    def __init__(self):
        self.clf = RandomForestClassifier(n_estimators=30, random_state=42)
        self._train_baseline()

    def _train_baseline(self):
        np.random.seed(42)
        # Normal physiologic profiles
        hr_norm = np.random.normal(72, 8, 200)
        sdnn_norm = np.random.normal(50, 10, 200)
        rmssd_norm = np.random.normal(40, 8, 200)
        x_norm = np.column_stack((hr_norm, sdnn_norm, rmssd_norm))
        y_norm = np.zeros(200)

        # Acute cardiac / ischemia profiles
        hr_acute = np.random.normal(135, 15, 200)
        sdnn_acute = np.random.normal(10, 4, 200)
        rmssd_acute = np.random.normal(8, 3, 200)
        x_acute = np.column_stack((hr_acute, sdnn_acute, rmssd_acute))
        y_acute = np.ones(200)

        x = np.vstack((x_norm, x_acute))
        y = np.concatenate((y_norm, y_acute))
        self.clf.fit(x, y)

    def evaluate_risk(self, metrics):
        vec = np.array([[metrics['hr_bpm'], metrics['sdnn_ms'], metrics['rmssd_ms']]])
        return self.clf.predict_proba(vec)[0][1]


class MonitorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Closed-Loop Telemetry & Micro-Actuator Monitor")
        self.root.geometry("1050x700")
        self.root.configure(bg="#1e1e1e")

        self.processor = BiosignalProcessor(sampling_rate=250)
        self.actuator = ClosedLoopActuator()
        self.model = DiagnosticModel()

        self.sim_mode = "NORMAL"
        self.is_running = True
        self.t_step = 0

        self._build_layout()
        self.update_simulation()

    def _build_layout(self):
        # Header Frame
        header = tk.Frame(self.root, bg="#2d2d2d", padx=15, pady=10)
        header.pack(fill=tk.X)

        title_lbl = tk.Label(header, text="Physiological Telemetry & Autonomous Drug Delivery Monitor", 
                             font=("Segoe UI", 13, "bold"), fg="#ffffff", bg="#2d2d2d")
        title_lbl.pack(side=tk.LEFT)

        self.mode_lbl = tk.Label(header, text="STATUS: NORMAL BASELINE", 
                                 font=("Segoe UI", 10, "bold"), fg="#4CAF50", bg="#2d2d2d")
        self.mode_lbl.pack(side=tk.RIGHT)

        # Main Container
        main_frame = tk.Frame(self.root, bg="#1e1e1e", padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left Column: Graph
        graph_frame = tk.Frame(main_frame, bg="#252526", padx=5, pady=5)
        graph_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(6, 5), dpi=100, facecolor='#252526')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#1e1e1e')
        self.ax.tick_params(colors='#cccccc', labelsize=8)
        self.ax.spines['bottom'].set_color('#444444')
        self.ax.spines['top'].set_color('#444444')
        self.ax.spines['left'].set_color('#444444')
        self.ax.spines['right'].set_color('#444444')
        self.ax.set_title("Real-Time Filtered ECG & R-Peak Detection", color="#ffffff", fontsize=10)
        self.ax.set_xlabel("Time (seconds)", color="#aaaaaa", fontsize=8)
        self.ax.set_ylabel("Amplitude (mV)", color="#aaaaaa", fontsize=8)

        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Right Column: Controls & Indicators
        panel_frame = tk.Frame(main_frame, bg="#252526", width=360, padx=12, pady=10)
        panel_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))

        # Simulation Mode Buttons
        btn_box = tk.LabelFrame(panel_frame, text="Simulation Controls", font=("Segoe UI", 9, "bold"), 
                                fg="#cccccc", bg="#252526", padx=8, pady=8)
        btn_box.pack(fill=tk.X, pady=(0, 10))

        btn_normal = tk.Button(btn_box, text="Set Normal Baseline", bg="#3a3a3a", fg="#ffffff", 
                               relief=tk.FLAT, font=("Segoe UI", 9), command=self._set_normal)
        btn_normal.pack(fill=tk.X, pady=3)

        btn_acute = tk.Button(btn_box, text="Simulate Acute Cardiac Event", bg="#b71c1c", fg="#ffffff", 
                              relief=tk.FLAT, font=("Segoe UI", 9, "bold"), command=self._set_acute)
        btn_acute.pack(fill=tk.X, pady=3)

        # Vitals Telemetry Panel
        vitals_box = tk.LabelFrame(panel_frame, text="Physiological Metrics", font=("Segoe UI", 9, "bold"), 
                                   fg="#cccccc", bg="#252526", padx=8, pady=8)
        vitals_box.pack(fill=tk.X, pady=(0, 10))

        self.lbl_hr = self._add_stat(vitals_box, "Heart Rate:", "72 BPM")
        self.lbl_sdnn = self._add_stat(vitals_box, "SDNN (HRV):", "52.4 ms")
        self.lbl_spo2 = self._add_stat(vitals_box, "SpO2 Saturation:", "98 %")
        self.lbl_glucose = self._add_stat(vitals_box, "Glucose Level:", "105 mg/dL")
        self.lbl_risk = self._add_stat(vitals_box, "Cardiac Risk Score:", "0.02")

        # Actuator Status Panel
        act_box = tk.LabelFrame(panel_frame, text="Micro-Valve Actuator Control", font=("Segoe UI", 9, "bold"), 
                                fg="#cccccc", bg="#252526", padx=8, pady=8)
        act_box.pack(fill=tk.X, pady=(0, 10))

        self.lbl_valve = self._add_stat(act_box, "Valve Aperture:", "0.0 %")
        self.lbl_flow = self._add_stat(act_box, "Infusion Flow Rate:", "0.0 uL/min")
        self.lbl_status = self._add_stat(act_box, "Delivery Status:", "STANDBY")

        # Clinical & Dietary Guidance
        guidance_box = tk.LabelFrame(panel_frame, text="Metabolic & Dietary Evaluation", font=("Segoe UI", 9, "bold"), 
                                     fg="#cccccc", bg="#252526", padx=8, pady=8)
        guidance_box.pack(fill=tk.BOTH, expand=True)

        self.txt_guidance = tk.Text(guidance_box, height=5, bg="#1e1e1e", fg="#e0e0e0", 
                                    font=("Consolas", 8), relief=tk.FLAT, wrap=tk.WORD)
        self.txt_guidance.pack(fill=tk.BOTH, expand=True, pady=4)

    def _add_stat(self, parent, label_text, default_val):
        frame = tk.Frame(parent, bg="#252526")
        frame.pack(fill=tk.X, pady=2)
        lbl_title = tk.Label(frame, text=label_text, font=("Segoe UI", 8), fg="#aaaaaa", bg="#252526")
        lbl_title.pack(side=tk.LEFT)
        lbl_val = tk.Label(frame, text=default_val, font=("Segoe UI", 8, "bold"), fg="#ffffff", bg="#252526")
        lbl_val.pack(side=tk.RIGHT)
        return lbl_val

    def _set_normal(self):
        self.sim_mode = "NORMAL"
        self.mode_lbl.config(text="STATUS: NORMAL BASELINE", fg="#4CAF50")

    def _set_acute(self):
        self.sim_mode = "ACUTE"
        self.mode_lbl.config(text="STATUS: ACUTE CARDIAC EVENT", fg="#f44336")

    def _generate_ecg_chunk(self):
        fs = 250
        duration = 3.0
        t = np.linspace(0, duration, int(fs * duration))
        
        if self.sim_mode == "NORMAL":
            hr_target = 72.0
            noise_amp = 0.05
            qrs_height = 1.2
            spo2 = 98.0 + np.random.uniform(-1, 1)
            glucose = 105.0 + np.random.uniform(-3, 3)
        else:
            hr_target = 140.0
            noise_amp = 0.20
            qrs_height = 0.7
            spo2 = 88.0 + np.random.uniform(-2, 1)
            glucose = 165.0 + np.random.uniform(-5, 5)

        freq = hr_target / 60.0
        # Synthetic ECG generation
        p_wave = 0.15 * np.sin(2 * np.pi * freq * t - 0.3)
        qrs = qrs_height * np.exp(-((np.sin(np.pi * freq * t) ** 2) / 0.01))
        t_wave = 0.25 * np.sin(2 * np.pi * freq * t + 0.5)
        noise = np.random.normal(0, noise_amp, len(t))
        raw_ecg = p_wave + qrs + t_wave + noise

        return t, raw_ecg, spo2, glucose

    def update_simulation(self):
        t, raw_ecg, spo2, glucose = self._generate_ecg_chunk()
        filtered_ecg, peaks, metrics = self.processor.extract_features(raw_ecg)
        risk = self.model.evaluate_risk(metrics)
        actuation = self.actuator.compute_actuation(risk)

        # Plot ECG
        self.ax.clear()
        self.ax.set_facecolor('#1e1e1e')
        self.ax.plot(t, filtered_ecg, color='#00e5ff', lw=1.2, label='Filtered ECG')
        if len(peaks) > 0:
            self.ax.plot(t[peaks], filtered_ecg[peaks], "ro", markersize=4, label='R-Peaks')
        self.ax.set_title("Real-Time Filtered ECG & Peak Detection", color="#ffffff", fontsize=9)
        self.ax.set_xlabel("Time (s)", color="#aaaaaa", fontsize=8)
        self.ax.set_ylabel("mV", color="#aaaaaa", fontsize=8)
        self.ax.tick_params(colors='#cccccc', labelsize=8)
        self.ax.grid(True, color='#333333', linestyle='--', linewidth=0.5)
        self.ax.set_ylim(-0.8, 1.8)
        self.canvas.draw()

        # Update Vitals
        self.lbl_hr.config(text=f"{metrics['hr_bpm']:.1f} BPM")
        self.lbl_sdnn.config(text=f"{metrics['sdnn_ms']:.1f} ms")
        self.lbl_spo2.config(text=f"{spo2:.1f} %", fg="#f44336" if spo2 < 92 else "#ffffff")
        self.lbl_glucose.config(text=f"{glucose:.1f} mg/dL", fg="#ff9800" if glucose > 140 else "#ffffff")
        
        risk_color = "#f44336" if risk > 0.6 else ("#ff9800" if risk > 0.25 else "#4caf50")
        self.lbl_risk.config(text=f"{risk:.3f}", fg=risk_color)

        # Update Actuator Panel
        self.lbl_valve.config(text=f"{actuation['valve_open_pct']:.1f} %", fg=risk_color)
        self.lbl_flow.config(text=f"{actuation['flow_rate_ul_min']:.1f} uL/min")
        self.lbl_status.config(text=actuation['status'], fg=risk_color)

        # Update Guidance Note
        self.txt_guidance.delete('1.0', tk.END)
        if risk > 0.5:
            self.txt_guidance.insert(tk.END, "[CRITICAL ACTION]: Autonomous micro-valve triggered. Emergency cardiac infusion initiated.\n")
            self.txt_guidance.insert(tk.END, "- Restrict acute sodium/fluid load.\n- Continuous airway and pulse oximetry monitoring.")
        else:
            self.txt_guidance.insert(tk.END, "[METABOLIC STATUS]: Physiologic profile stable.\n")
            if glucose > 120:
                self.txt_guidance.insert(tk.END, "- Glycemic trend elevated: recommend low-GI complex carbohydrate adjustment.")
            else:
                self.txt_guidance.insert(tk.END, "- Biomarkers within normative target ranges.")

        # Schedule next refresh (every 1.5 seconds)
        self.root.after(1500, self.update_simulation)


if __name__ == "__main__":
    root = tk.Tk()
    app = MonitorGUI(root)
    root.mainloop()
