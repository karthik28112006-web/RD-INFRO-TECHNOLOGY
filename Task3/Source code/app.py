import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

SYMPTOMS = [
    "chest_pain", "shortness_of_breath", "fatigue", "irregular_heartbeat",
    "dizziness", "high_blood_pressure", "swelling_legs", "cold_sweats",
    "excessive_thirst", "frequent_urination", "blurred_vision",
    "slow_healing_wounds", "unexplained_weight_loss", "numbness_hands_feet",
    "increased_hunger"
]

HEART_SYMPTOMS = {
    "chest_pain", "shortness_of_breath", "irregular_heartbeat", "dizziness",
    "high_blood_pressure", "swelling_legs", "cold_sweats", "fatigue"
}

DIABETES_SYMPTOMS = {
    "excessive_thirst", "frequent_urination", "blurred_vision",
    "slow_healing_wounds", "unexplained_weight_loss", "numbness_hands_feet",
    "increased_hunger", "fatigue"
}


def generate_mock_dataset(n_samples=1200, seed=42):
    rng = np.random.default_rng(seed)
    data = rng.integers(0, 2, size=(n_samples, len(SYMPTOMS)))
    df = pd.DataFrame(data, columns=SYMPTOMS)

    heart_idx = [SYMPTOMS.index(s) for s in HEART_SYMPTOMS]
    diabetes_idx = [SYMPTOMS.index(s) for s in DIABETES_SYMPTOMS]

    heart_score = df.iloc[:, heart_idx].sum(axis=1)
    diabetes_score = df.iloc[:, diabetes_idx].sum(axis=1)

    heart_noise = rng.normal(0, 0.6, n_samples)
    diabetes_noise = rng.normal(0, 0.6, n_samples)

    heart_label = ((heart_score + heart_noise) >= 4).astype(int)
    diabetes_label = ((diabetes_score + diabetes_noise) >= 4).astype(int)

    df["heart_disease"] = heart_label
    df["diabetes"] = diabetes_label
    return df


@st.cache_resource
def train_models():
    df = generate_mock_dataset()
    X = df[SYMPTOMS]
    results = {}

    for target in ["heart_disease", "diabetes"]:
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            random_state=42,
            class_weight="balanced"
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"[INIT] {target} model trained | Test Accuracy: {acc:.4f}")

        results[target] = {
            "model": model,
            "accuracy": acc,
            "importances": pd.Series(
                model.feature_importances_, index=SYMPTOMS
            ).sort_values(ascending=False)
        }

    return results


def risk_label(prob, threshold=0.5):
    return "High Risk" if prob >= threshold else "Low Risk"


def main():
    st.set_page_config(page_title="Multi-Disease Risk Predictor", layout="wide")
    st.title("🩺 Multi-Disease Risk Prediction System")
    st.caption("Random Forest classifiers trained on synthetic symptom data. For demonstration purposes only — not medical advice.")

    models = train_models()

    st.sidebar.header("Model Accuracy (Test Set)")
    for target, res in models.items():
        st.sidebar.metric(target.replace("_", " ").title(), f"{res['accuracy']*100:.2f}%")

    st.header("Patient Symptom Checklist")
    st.write("Select all symptoms the patient currently experiences:")

    cols = st.columns(3)
    symptom_values = {}
    for i, symptom in enumerate(SYMPTOMS):
        col = cols[i % 3]
        label = symptom.replace("_", " ").title()
        symptom_values[symptom] = col.checkbox(label, key=symptom)

    input_vector = pd.DataFrame(
        [[int(symptom_values[s]) for s in SYMPTOMS]], columns=SYMPTOMS
    )

    st.divider()

    if st.button("Predict Disease Risk", type="primary"):
        st.header("Prediction Results")
        result_cols = st.columns(len(models))

        for col, (target, res) in zip(result_cols, models.items()):
            model = res["model"]
            prob = model.predict_proba(input_vector)[0][1]
            label = risk_label(prob)

            with col:
                st.subheader(target.replace("_", " ").title())
                if label == "High Risk":
                    st.error(f"⚠️ {label} — {prob*100:.1f}% probability")
                else:
                    st.success(f"✅ {label} — {prob*100:.1f}% probability")
                st.progress(min(int(prob * 100), 100))

        st.divider()
        st.subheader("Top Contributing Symptoms")
        imp_cols = st.columns(len(models))
        for col, (target, res) in zip(imp_cols, models.items()):
            with col:
                st.write(f"**{target.replace('_', ' ').title()}**")
                st.bar_chart(res["importances"].head(6))
    else:
        st.info("Select symptoms above and click 'Predict Disease Risk' to see results.")


if __name__ == "__main__":
    main()
