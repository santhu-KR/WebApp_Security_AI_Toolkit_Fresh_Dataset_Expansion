from pathlib import Path
import joblib

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "security_model_v1.joblib"
)

print("Loading V1 model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully!")

test_samples = [
    {
        "name": "Safe SQL Query",
        "code": """
        cursor.execute(
            "SELECT * FROM users WHERE id = ?",
            (user_id,)
        )
        """
    },
    {
        "name": "Potential SQL Injection",
        "code": """
        query = "SELECT * FROM users WHERE id = " + user_input
        cursor.execute(query)
        """
    },
    {
        "name": "Safe Password Hashing",
        "code": """
        hashed_password = bcrypt.hashpw(
            password.encode(),
            bcrypt.gensalt()
        )
        """
    }
]

for sample in test_samples:

    prediction = model.predict([sample["code"]])[0]

    probabilities = model.predict_proba(
        [sample["code"]]
    )[0]

    confidence = max(probabilities) * 100

    result = (
        "VULNERABLE"
        if prediction == 1
        else "SAFE"
    )

    print("\n------------------------------")
    print("Test:", sample["name"])
    print("Prediction:", result)
    print(f"Confidence: {confidence:.2f}%")