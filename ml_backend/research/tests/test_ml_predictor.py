from ml_predictor import (
    predict_vulnerability,
    get_model_status
)


# ============================================================
# MODEL STATUS
# ============================================================

print("\nMODEL STATUS")

print(get_model_status())


# ============================================================
# TEST CASES
# ============================================================

test_cases = [

    "The application constructs SQL queries using unsanitized user input.",

    "The application allows users to access another user's account information by changing the user ID.",

    "The application executes operating system commands using user-controlled input.",

    "The web application reflects user input into HTML without proper encoding.",

    "The application allows requests to internal server resources.",

    "The application uses weak authentication and does not properly validate user sessions.",

    "The application allows directory traversal through file paths.",

    "The application uses insecure deserialization of untrusted data.",

    "The application exposes sensitive information in error messages.",

    "The application has insecure security configuration."

]


# ============================================================
# RUN PREDICTIONS
# ============================================================

print("\n")

print("=" * 60)

print("ML VULNERABILITY PREDICTION TEST")

print("=" * 60)


for index, description in enumerate(
    test_cases,
    start=1
):

    result = predict_vulnerability(
        description
    )


    print(f"\nTest {index}")

    print("-" * 40)

    print("Input:")

    print(description)


    if result["success"]:

        print("\nPredicted Category:")

        print(result["predicted_category"])

        print("\nAlgorithm:")

        print(result["algorithm"])

        print("\nDecision Score:")

        print(result["decision_score"])

    else:

        print("Prediction failed:")

        print(result["error"])


print("\n")

print("=" * 60)

print("TEST COMPLETED")

print("=" * 60)