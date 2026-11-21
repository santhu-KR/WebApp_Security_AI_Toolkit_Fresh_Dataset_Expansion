import os
import joblib

from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = CURRENT_DIR.parent

MODEL_DIRECTORY = (
    PROJECT_ROOT
    / "ml_backend"
    / "trained_models_v2"
)

MODEL_PATH = (
    MODEL_DIRECTORY
    / "best_vulnerability_model_v2.pkl"
)

VECTORIZER_PATH = (
    MODEL_DIRECTORY
    / "best_tfidf_vectorizer_v2.pkl"
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model = None

vectorizer = None

model_loaded = False


def load_model():

    global model
    global vectorizer
    global model_loaded

    try:

        if not MODEL_PATH.exists():

            print("ML MODEL ERROR: Model file not found")

            print(MODEL_PATH)

            return False


        if not VECTORIZER_PATH.exists():

            print("ML MODEL ERROR: Vectorizer file not found")

            print(VECTORIZER_PATH)

            return False


        model = joblib.load(MODEL_PATH)

        vectorizer = joblib.load(VECTORIZER_PATH)

        model_loaded = True


        print("=" * 50)

        print("ML MODEL LOADED SUCCESSFULLY")

        print("Algorithm: Linear SVM")

        print("Version: V2")

        print("Model path:", MODEL_PATH)

        print("=" * 50)


        return True


    except Exception as error:

        model_loaded = False

        print("Failed to load ML model:")

        print(error)

        return False


# ============================================================
# PREDICT VULNERABILITY CATEGORY
# ============================================================

def predict_vulnerability(text):

    if not model_loaded:

        return {

            "success": False,

            "error": "ML model is not loaded"

        }


    if not text or not isinstance(text, str):

        return {

            "success": False,

            "error": "Valid vulnerability description is required"

        }


    try:

        # Convert text into TF-IDF features

        features = vectorizer.transform([text])


        # Predict category

        prediction = model.predict(features)[0]


        # Get SVM decision scores
        # These are scores, NOT probabilities.

        decision_scores = model.decision_function(features)


        decision_score = None


        if len(decision_scores.shape) == 2:

            class_scores = decision_scores[0]

            predicted_index = list(model.classes_).index(
                prediction
            )

            decision_score = float(
                class_scores[predicted_index]
            )


        elif len(decision_scores.shape) == 1:

            decision_score = float(
                decision_scores[0]
            )


        return {

            "success": True,

            "predicted_category": str(prediction),

            "algorithm": "Linear SVM",

            "model_version": "V2",

            "decision_score": decision_score,

            "score_type": "SVM decision score, not probability"

        }


    except Exception as error:

        return {

            "success": False,

            "error": str(error)

        }


# ============================================================
# PREDICT MULTIPLE FINDINGS
# ============================================================

def predict_multiple(findings):

    results = []


    for finding in findings:

        if isinstance(finding, dict):

            description = (
                finding.get("description")
                or finding.get("title")
                or finding.get("name")
                or ""
            )

        else:

            description = str(finding)


        prediction = predict_vulnerability(
            description
        )


        results.append({

            "finding": finding,

            "prediction": prediction

        })


    return results


# ============================================================
# GET MODEL STATUS
# ============================================================

def get_model_status():

    return {

        "loaded": model_loaded,

        "algorithm": "Linear SVM",

        "version": "V2",

        "model_file": str(MODEL_PATH),

        "vectorizer_file": str(VECTORIZER_PATH)

    }


# ============================================================
# LOAD MODEL AUTOMATICALLY
# ============================================================

load_model()