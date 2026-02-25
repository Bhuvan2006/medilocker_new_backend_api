from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import os
import json
import traceback
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file in project root
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(project_root, ".env")
load_dotenv(env_path)

# -----------------------------
# Load model (Docker-safe path)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "diabetes_model.pkl")

model = joblib.load(MODEL_PATH)

app = FastAPI(title="Diabetes Prediction API")

# -----------------------------
# CORS (allow frontend dev server)
# -----------------------------
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "diabetes_model.pkl")
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Diabetes Prediction API")

# -----------------------------
# Allow frontend (React) to call backend
# -----------------------------
origins = [
    "http://localhost:8080",  # your React dev server
    "http://127.0.0.1:8080",   # sometimes React runs here
    "http://localhost:8081",  # your React dev server (alternative)
    "http://127.0.0.1:8081"   # sometimes React runs here
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,        # list of allowed origins
    allow_credentials=True,
    allow_methods=["*"],          # allow all HTTP methods
    allow_headers=["*"],          # allow all headers
)

# -----------------------------
# Health check
# -----------------------------
@app.get("/")
def health():
    return {"message": "Diabetes model API is running"}

# -----------------------------
# Request Schema
# -----------------------------
class PredictRequest(BaseModel):
    pregnancies: float
    glucose: float
    bloodPressure: float
    skinThickness: float
    insulin: float
    bmi: float
    diabetesPedigree: float
    age: float

# ─────────────────────────────────────────────────────────────
# AI / LLM Setup  (must be defined BEFORE the /predict route)
# ─────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("[WARNING] GEMINI_API_KEY not set in app/.env — LLM explanations will be disabled")

genai_model = None
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    genai_model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        generation_config={"temperature": 0.2}
    )


SYSTEM_PROMPT = """
You are a clinical decision support assistant.

You must explain a diabetes risk assessment using ONLY the structured data provided.
You are NOT allowed to:
- Add new risk factors
- Add medical advice
- Diagnose disease
- Infer causes not explicitly mentioned
- Mention model internals or algorithms

You MUST:
- Use cautious, non-diagnostic language
- Explain why the risk level was assigned
- Reference top_factors, pattern_detected, and counterfactual
- Write for clinicians (clear, concise, neutral tone)

If information is missing, do NOT speculate.
"""

def build_user_prompt(reasoning: dict) -> str:
    return f"""
Using the following structured risk assessment data, write a short clinician-friendly explanation.

Risk Assessment (GROUND TRUTH):
- Risk Level       : {reasoning.get("risk_level")}
- Confidence       : {reasoning.get("confidence")}
- Pattern Detected : {reasoning.get("pattern_detected")}
- Counterfactual   : {reasoning.get("counterfactual")}
- Top Factors      : {json.dumps(reasoning.get("top_factors", []), indent=2)}
based on the patient's input data and the model's analysis elaborate it upto 10-15 sentenses in a clear and neutral tone, 
-within the 10-15 lines tell some lifestyle habits that affect his disease with misinformation.
Do not introduce new information. Stay within the data above.
"""

def generate_explanation(reasoning: dict) -> str:
    if genai_model is None:
        return "LLM explanation unavailable — check GEMINI_API_KEY in app/.env"
    try:
        prompt = f"{SYSTEM_PROMPT}\n\n{build_user_prompt(reasoning)}"
        response = genai_model.generate_content(prompt)
        return response.text
    except Exception as e:
        traceback.print_exc()
        error_msg = str(e)
        # Check for invalid API key
        if "API key not valid" in error_msg or "API_KEY_INVALID" in error_msg:
            print(f"[Gemini Error] Invalid API key - LLM explanations disabled")
            return "LLM explanation unavailable — check GEMINI_API_KEY in app/.env"
        print(f"[Gemini Error] {e}")
        return f"AI explanation unavailable. ({type(e).__name__}: {e})"

# ─────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────

def calculate_confidence(risk_percentage: float) -> float:
    """Scale 0–100 risk percentage → 0.60–0.95 confidence score."""
    return round(0.6 + (risk_percentage / 100) * 0.35, 2)

def extract_top_factors(shap_values: list, top_k: int = 3) -> list:
    """Sort SHAP values by impact descending and return top-k."""
    sorted_factors = sorted(shap_values, key=lambda x: x["value"], reverse=True)
    return [
        {
            "feature": factor["feature"].lower().replace(" ", "_"),
            "impact": round(factor["value"], 2)
        }
        for factor in sorted_factors[:top_k]
    ]

def detect_pattern(input_data: dict) -> str:
    patterns = []
    if input_data["glucose"] > 140:
        patterns.append("elevated glucose")
    if input_data["bmi"] >= 30:
        patterns.append("high BMI")
    if input_data["diabetesPedigree"] > 0.6:
        patterns.append("strong family history")
    if patterns:
        return f"{', '.join(patterns).capitalize()} — higher diabetes risk group"
    return "No strong high-risk metabolic patterns detected"

def generate_counterfactual(top_factors: list) -> str:
    if not top_factors:
        return "Insufficient factor data for counterfactual reasoning"
    strongest = top_factors[0]["feature"]
    return f"Reducing {strongest.replace('_', ' ')} has the strongest impact on lowering risk"

def build_reasoning_json(model_output: dict, input_data: dict) -> dict:
    top_factors = extract_top_factors(model_output["shapValues"])
    return {
        "risk_level": model_output["riskLevel"],
        "confidence": calculate_confidence(model_output["riskPercentage"]),
        "top_factors": top_factors,
        "pattern_detected": detect_pattern(input_data),
        "counterfactual": generate_counterfactual(top_factors)
    }

# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────

@app.get("/")
def health():
    return {"message": "Diabetes model API is running"}

@app.post("/predict")
def predict(data: PredictRequest):

    features = np.array([[
        data.pregnancies,
        data.glucose,
        data.bloodPressure,
        data.skinThickness,
        data.insulin,
        data.bmi,
        data.diabetesPedigree,
        data.age
    ]])

    probability = model.predict_proba(features)[0][1]
    risk_percentage = int(probability * 100)

    risk_level = (
        "Low"      if probability < 0.30 else
        "Moderate" if probability < 0.60 else
        "High"
    )

    # SHAP-style feature importance values
    feature_names = [
        "Pregnancies", "Glucose", "Blood Pressure",
        "Skin Thickness", "Insulin", "BMI",
        "Diabetes Pedigree", "Age"
    ]
    importances = model.feature_importances_
    shap_values = [
        {"feature": name, "value": float(imp)}
        for name, imp in zip(feature_names, importances)
    ]

    response_from_model = {
        "riskLevel": risk_level,
        "riskPercentage": risk_percentage,
        "shapValues": shap_values
    }
    patient_input = {
        "glucose": data.glucose,
        "bmi": data.bmi,
        "diabetesPedigree": data.diabetesPedigree,
        "age": data.age
    }

    reasoning = build_reasoning_json(
        model_output=response_from_model,
        input_data=patient_input
    )

    explanation = generate_explanation(reasoning)

    return {
        "riskProbability": round(float(probability), 3),
        "riskPercentage": risk_percentage,
        "riskLevel": risk_level,
        "confidence": reasoning["confidence"],
        "top_factors": reasoning["top_factors"],
        "pattern_detected": reasoning["pattern_detected"],
        "counterfactual": reasoning["counterfactual"],
        "explanation": explanation
    }
