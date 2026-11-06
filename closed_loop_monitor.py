import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


class BiosignalProcessor:
    def __init__(self, fs=250):
        self.fs = fs

    def bandpass_filter(self, signal_data, lowcut=0.5, highcut=40.0, order=4):
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype="band")
        return filtfilt(b, a, signal_data)

    def extract_ecg_features(self, raw_ecg):
        filtered = self.bandpass_filter(raw_ecg)
        min_distance = int(0.3 * self.fs)
        peaks, _ = find_peaks(filtered, distance=min_distance, height=np.std(filtered) * 1.2)

        if len(peaks) < 2:
            return {"hr_bpm": 0.0, "sdnn_ms": 0.0, "rmssd_ms": 0.0}

        rr_intervals = np.diff(peaks) / self.fs * 1000.0
        hr_bpm = 60000.0 / np.mean(rr_intervals)
        sdnn_ms = np.std(rr_intervals)
        rmssd_ms = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))

        return {
            "hr_bpm": round(float(hr_bpm), 2),
            "sdnn_ms": round(float(sdnn_ms), 2),
            "rmssd_ms": round(float(rmssd_ms), 2),
        }


class ProportionalValveActuator:
    def __init__(self, kp=1.4, ki=0.05, max_flow_rate_ul_min=50.0):
        self.kp = kp
        self.ki = ki
        self.max_flow_rate = max_flow_rate_ul_min
        self.integral_error = 0.0

    def compute_actuation(self, risk_probability, spo2_level, target_spo2=96.0):
        if risk_probability < 0.65:
            self.integral_error = 0.0
            return {"valve_open_pct": 0.0, "flow_rate_ul_min": 0.0, "status": "STANDBY"}

        spo2_deficit = max(0.0, target_spo2 - spo2_level)
        error = (risk_probability * 10.0) + spo2_deficit
        self.integral_error += error

        control_output = (self.kp * error) + (self.ki * self.integral_error)
        valve_pct = float(np.clip(control_output * 5.0, 0.0, 100.0))
        flow_rate = round((valve_pct / 100.0) * self.max_flow_rate, 2)

        return {
            "valve_open_pct": round(valve_pct, 1),
            "flow_rate_ul_min": flow_rate,
            "status": "ACTIVE_DELIVERY",
        }


class MetabolicDietAnalyzer:
    def evaluate_profile(self, glucose_mg_dl, triglycerides_mg_dl, sleep_hours, spo2_pct):
        recommendations = []
        metabolic_state = "NORMAL"

        if glucose_mg_dl >= 140.0:
            metabolic_state = "ELEVATED_GLYCEMIC"
            recommendations.append("Reduce simple carbohydrate intake; prioritize low-glycemic fiber.")
        elif glucose_mg_dl < 70.0:
            metabolic_state = "HYPOGLYCEMIC_ALERT"
            recommendations.append("Immediate intake of 15g fast-acting carbohydrates required.")

        if triglycerides_mg_dl > 150.0:
            if metabolic_state == "NORMAL":
                metabolic_state = "DYSLIPIDEMIC"
            recommendations.append("Limit saturated fats; increase omega-3 fatty acid ratio.")

        if sleep_hours < 6.0:
            recommendations.append("Sleep deficit detected; adjust evening caloric timing to stabilize cortisol.")

        if spo2_pct < 94.0:
            recommendations.append("Low nocturnal/resting SpO2; monitor respiratory efficiency.")

        if not recommendations:
            recommendations.append("Metabolic biomarkers within target range. Maintain current macronutrient balance.")

        return {
            "metabolic_state": metabolic_state,
            "dietary_adjustments": recommendations,
        }


def build_synthetic_dataset(n_samples=600, random_state=42):
    rng = np.random.default_rng(random_state)
    # Features: [hr_bpm, sdnn_ms, rmssd_ms, spo2_pct, glucose_mg_dl, triglycerides_mg_dl]
    normal_hr = rng.normal(72, 8, n_samples // 2)
    normal_sdnn = rng.normal(55, 12, n_samples // 2)
    normal_rmssd = rng.normal(42, 10, n_samples // 2)
    normal_spo2 = rng.normal(97.5, 1.0, n_samples // 2)
    normal_gluc = rng.normal(95, 12, n_samples // 2)
    normal_trig = rng.normal(115, 20, n_samples // 2)

    acute_hr = rng.normal(135, 18, n_samples // 2)
    acute_sdnn = rng.normal(18, 6, n_samples // 2)
    acute_rmssd = rng.normal(12, 4, n_samples // 2)
    acute_spo2 = rng.normal(90.5, 2.5, n_samples // 2)
    acute_gluc = rng.normal(155, 25, n_samples // 2)
    acute_trig = rng.normal(190, 35, n_samples // 2)

    X_normal = np.column_stack([normal_hr, normal_sdnn, normal_rmssd, normal_spo2, normal_gluc, normal_trig])
    X_acute = np.column_stack([acute_hr, acute_sdnn, acute_rmssd, acute_spo2, acute_gluc, acute_trig])

    X = np.vstack([X_normal, X_acute])
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))
    return X, y


def simulate_ecg_waveform(duration_sec=10, fs=250, heart_rate_bpm=75.0, noise_level=0.05):
    t = np.linspace(0, duration_sec, duration_sec * fs)
    rr_sec = 60.0 / heart_rate_bpm
    ecg = np.zeros_like(t)
    beat_times = np.arange(0.5, duration_sec, rr_sec)

    for bt in beat_times:
        ecg += 1.2 * np.exp(-((t - bt) ** 2) / (2 * (0.012 ** 2)))
        ecg -= 0.2 * np.exp(-((t - (bt - 0.03)) ** 2) / (2 * (0.01 ** 2)))
        ecg += 0.25 * np.exp(-((t - (bt + 0.18)) ** 2) / (2 * (0.04 ** 2)))

    baseline_drift = 0.15 * np.sin(2 * np.pi * 0.2 * t)
    noise = np.random.normal(0, noise_level, size=len(t))
    return ecg + baseline_drift + noise


if __name__ == "__main__":
    X, y = build_synthetic_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    print("Model Validation Report:")
    print(classification_report(y_test, preds, target_names=["Stable", "Acute_Cardiac_Event"]))

    processor = BiosignalProcessor(fs=250)
    valve = ProportionalValveActuator(kp=1.4, ki=0.05, max_flow_rate_ul_min=50.0)
    metabolic_analyzer = MetabolicDietAnalyzer()

    raw_ecg_critical = simulate_ecg_waveform(duration_sec=10, fs=250, heart_rate_bpm=142.0, noise_level=0.04)
    ecg_metrics = processor.extract_ecg_features(raw_ecg_critical)

    patient_Telemetry = {
        "spo2_pct": 89.5,
        "glucose_mg_dl": 162.0,
        "triglycerides_mg_dl": 185.0,
        "sleep_hours": 5.2,
    }

    feature_vector = np.array([[
        ecg_metrics["hr_bpm"],
        ecg_metrics["sdnn_ms"],
        ecg_metrics["rmssd_ms"],
        patient_Telemetry["spo2_pct"],
        patient_Telemetry["glucose_mg_dl"],
        patient_Telemetry["triglycerides_mg_dl"],
    ]])

    risk_prob = float(clf.predict_proba(feature_vector)[0][1])
    actuation_state = valve.compute_actuation(risk_prob, patient_Telemetry["spo2_pct"])
    metabolic_report = metabolic_analyzer.evaluate_profile(
        patient_Telemetry["glucose_mg_dl"],
        patient_Telemetry["triglycerides_mg_dl"],
        patient_Telemetry["sleep_hours"],
        patient_Telemetry["spo2_pct"],
    )

    print("Telemetry & Closed-Loop Control Output:")
    print(f"ECG Metrics: {ecg_metrics}")
    print(f"Cardiac Risk Probability: {risk_prob:.3f}")
    print(f"Micro-Valve Actuator: {actuation_state}")
    print(f"Metabolic & Dietary Assessment: {metabolic_report}")