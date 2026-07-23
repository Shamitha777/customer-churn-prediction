from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "churn_model.pkl")


model = joblib.load(MODEL_PATH)

FEATURE_COLUMNS = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "gender_Male",
    "Partner_Yes",
    "Dependents_Yes",
    "PhoneService_Yes",
    "MultipleLines_No phone service",
    "MultipleLines_Yes",
    "InternetService_Fiber optic",
    "InternetService_No",
    "OnlineSecurity_No internet service",
    "OnlineSecurity_Yes",
    "OnlineBackup_No internet service",
    "OnlineBackup_Yes",
    "DeviceProtection_No internet service",
    "DeviceProtection_Yes",
    "TechSupport_No internet service",
    "TechSupport_Yes",
    "StreamingTV_No internet service",
    "StreamingTV_Yes",
    "StreamingMovies_No internet service",
    "StreamingMovies_Yes",
    "Contract_One year",
    "Contract_Two year",
    "PaperlessBilling_Yes",
    "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic check",
    "PaymentMethod_Mailed check",
]


coef = model.coef_[0]

def pretty_label(feature_name):
    mapping = {
        "InternetService_Fiber optic": "Internet Service: Fiber Optic",
        "Contract_One year": "One-Year Contract",
        "Contract_Two year": "Two-Year Contract",
        "MonthlyCharges": "Monthly Charges",
        "TotalCharges": "Total Charges",
    }
    if feature_name in mapping:
        return mapping[feature_name]
    label = feature_name.replace("_", " ")
    # Normalize common tokens
    label = label.replace("  ", " ")
    return label.title()

FEATURE_IMPORTANCE = sorted(
    [
        {
            "name": name,
            "label": pretty_label(name),
            "score": abs(value),
            "sign": "positive" if value > 0 else "negative",
            "direction": "Increases churn" if value > 0 else "Reduces churn",
        }
        for name, value in zip(FEATURE_COLUMNS, coef)
    ],
    key=lambda item: item["score"],
    reverse=True,
)[:6]

max_score = FEATURE_IMPORTANCE[0]["score"] if FEATURE_IMPORTANCE else 1
for item in FEATURE_IMPORTANCE:
    item["strength"] = round(100 * item["score"] / max_score)

OPTION_SETS = {
    "gender": ["Female", "Male"],
    "partner": ["No", "Yes"],
    "dependents": ["No", "Yes"],
    "senior_citizen": ["0", "1"],
    "phone_service": ["No", "Yes"],
    "multiple_lines": ["No", "No phone service", "Yes"],
    "internet_service": ["DSL", "Fiber optic", "No"],
    "online_security": ["No", "No internet service", "Yes"],
    "online_backup": ["No", "No internet service", "Yes"],
    "device_protection": ["No", "No internet service", "Yes"],
    "tech_support": ["No", "No internet service", "Yes"],
    "streaming_tv": ["No", "No internet service", "Yes"],
    "streaming_movies": ["No", "No internet service", "Yes"],
    "contract": ["Month-to-month", "One year", "Two year"],
    "paperless_billing": ["No", "Yes"],
    "payment_method": [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ],
}


def build_feature_vector(form):
    senior_citizen = int(form.get("senior_citizen", "0"))
    tenure = float(form.get("tenure", 0))
    monthly_charges = float(form.get("monthly", 0))
    total_charges = float(form.get("total_charges", tenure * monthly_charges))

    features = {
        "SeniorCitizen": senior_citizen,
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "gender_Male": 1 if form.get("gender", "Female") == "Male" else 0,
        "Partner_Yes": 1 if form.get("partner", "No") == "Yes" else 0,
        "Dependents_Yes": 1 if form.get("dependents", "No") == "Yes" else 0,
        "PhoneService_Yes": 1 if form.get("phone_service", "No") == "Yes" else 0,
        "MultipleLines_No phone service": 1 if form.get("multiple_lines") == "No phone service" else 0,
        "MultipleLines_Yes": 1 if form.get("multiple_lines") == "Yes" else 0,
        "InternetService_Fiber optic": 1 if form.get("internet_service") == "Fiber optic" else 0,
        "InternetService_No": 1 if form.get("internet_service") == "No" else 0,
        "OnlineSecurity_No internet service": 1 if form.get("online_security") == "No internet service" else 0,
        "OnlineSecurity_Yes": 1 if form.get("online_security") == "Yes" else 0,
        "OnlineBackup_No internet service": 1 if form.get("online_backup") == "No internet service" else 0,
        "OnlineBackup_Yes": 1 if form.get("online_backup") == "Yes" else 0,
        "DeviceProtection_No internet service": 1 if form.get("device_protection") == "No internet service" else 0,
        "DeviceProtection_Yes": 1 if form.get("device_protection") == "Yes" else 0,
        "TechSupport_No internet service": 1 if form.get("tech_support") == "No internet service" else 0,
        "TechSupport_Yes": 1 if form.get("tech_support") == "Yes" else 0,
        "StreamingTV_No internet service": 1 if form.get("streaming_tv") == "No internet service" else 0,
        "StreamingTV_Yes": 1 if form.get("streaming_tv") == "Yes" else 0,
        "StreamingMovies_No internet service": 1 if form.get("streaming_movies") == "No internet service" else 0,
        "StreamingMovies_Yes": 1 if form.get("streaming_movies") == "Yes" else 0,
        "Contract_One year": 1 if form.get("contract") == "One year" else 0,
        "Contract_Two year": 1 if form.get("contract") == "Two year" else 0,
        "PaperlessBilling_Yes": 1 if form.get("paperless_billing") == "Yes" else 0,
        "PaymentMethod_Credit card (automatic)": 1 if form.get("payment_method") == "Credit card (automatic)" else 0,
        "PaymentMethod_Electronic check": 1 if form.get("payment_method") == "Electronic check" else 0,
        "PaymentMethod_Mailed check": 1 if form.get("payment_method") == "Mailed check" else 0,
    }

    return np.array([[features[col] for col in FEATURE_COLUMNS]])


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""
    churn_probability = None
    confidence = None
    risk_level = ""
    recommended_action = ""
    form_values = {}
    is_churn = False

    if request.method == "POST":
        try:
            data = build_feature_vector(request.form)
            prediction_value = model.predict(data)[0]
            is_churn = prediction_value == 1
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(data)[0]
                churn_probability = round(100 * proba[1], 1)
                confidence = round(100 * proba[prediction_value], 1)
                risk_level = "High 🔴" if churn_probability >= 50 else "Low 🟢"

            prediction = (
                "Customer Will Churn" if is_churn else "Customer Will Not Churn"
            )
            recommended_action = (
                "Offer a personalized retention package and proactive support."
                if is_churn
                else "Maintain service quality and monitor customer engagement."
            )
        except Exception as exc:
            prediction = f"Error making a prediction: {exc}"
            recommended_action = ""

        form_values = request.form

    return render_template(
        "index.html",
        prediction=prediction,
        churn_probability=churn_probability,
        confidence=confidence,
        risk_level=risk_level,
        recommended_action=recommended_action,
        is_churn=is_churn,
        feature_importance=FEATURE_IMPORTANCE,
        options=OPTION_SETS,
        form_values=form_values,
    )

if __name__ == "__main__":
    app.run(debug=True)