from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import joblib
import os

app = Flask(__name__)

print("Loading models...")
try:
    lr_model = joblib.load('models/logistic_regression_model.pkl')
    rf_model = joblib.load('models/random_forest_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    le_contract = joblib.load('models/encoder_contract.pkl')
    le_payment = joblib.load('models/encoder_payment.pkl')
    print("✓ All models loaded successfully!")
    print(f"✓ Expected features: {feature_names}")
except Exception as e:
    print(f"✗ Error: {e}")


def process_input(data):
    tenure = float(data['tenure'])
    monthly = float(data['monthly_charges'])
    total = float(data['total_charges'])
    support = float(data['support_calls'])
    late_pay = float(data['late_payments'])
    products = float(data['num_products'])
    partner_val = float(data['has_partner'])
    dep_val = float(data['has_dependents'])
    long_tenure = 1.0 if tenure > 24 else 0.0

    contract_enc = float(le_contract.transform([data['contract_type']])[0])
    payment_enc = float(le_payment.transform([data['payment_method']])[0])

    avg_spend = total / (tenure + 1)
    charge_ratio = monthly / (tenure + 1)
    high_value = 1.0 if monthly > 80 else 0.0
    new_customer = 1.0 if tenure < 6 else 0.0
    risk_score = support * 0.35 + late_pay * 0.45 + (1 - long_tenure) * 0.2
    loyalty = tenure * 0.4 + products * 0.3 + partner_val * 0.15 + dep_val * 0.15

    # Build as DataFrame with correct feature names
    feature_dict = {
        'age': [float(data['age'])],
        'tenure': [tenure],
        'monthly_charges': [monthly],
        'total_charges': [total],
        'num_products': [products],
        'has_internet': [float(data['has_internet'])],
        'has_phone': [float(data['has_phone'])],
        'is_senior': [float(data['is_senior'])],
        'has_partner': [partner_val],
        'has_dependents': [dep_val],
        'paperless_billing': [float(data['paperless_billing'])],
        'support_calls': [support],
        'late_payments': [late_pay],
        'contract_encoded': [contract_enc],
        'payment_encoded': [payment_enc],
        'avg_spend': [avg_spend],
        'charge_ratio': [charge_ratio],
        'high_value': [high_value],
        'long_tenure': [long_tenure],
        'new_customer': [new_customer],
        'risk_score': [risk_score],
        'loyalty': [loyalty],
    }

    df = pd.DataFrame(feature_dict)

    # Reorder columns to match training order
    df = df[feature_names]

    return df


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()

        print(f"Received data: {data}")

        features_df = process_input(data)
        print(f"Features shape: {features_df.shape}")
        print(f"Feature columns: {list(features_df.columns)}")

        features_scaled = scaler.transform(features_df)

        model_choice = data.get('model', 'random_forest')
        if model_choice == 'logistic_regression':
            model = lr_model
            model_name = "Logistic Regression"
        else:
            model = rf_model
            model_name = "Random Forest"

        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]

        churn_prob = float(probability[1])
        no_churn_prob = float(probability[0])

        if churn_prob >= 0.7:
            risk = "HIGH RISK"
            rec = "Immediate action! Offer special retention package."
        elif churn_prob >= 0.4:
            risk = "MEDIUM RISK"
            rec = "Monitor closely. Consider loyalty rewards."
        else:
            risk = "LOW RISK"
            rec = "Customer is stable. Continue regular engagement."

        result = {
            'status': 'success',
            'model_used': model_name,
            'prediction': int(prediction),
            'will_churn': bool(prediction == 1),
            'churn_probability': round(churn_prob * 100, 2),
            'no_churn_probability': round(no_churn_prob * 100, 2),
            'risk_level': risk,
            'recommendation': rec
        }
        print(f"Result: {result}")
        return jsonify(result)

    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print(f"ERROR: {error_msg}")
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})


@app.route('/model/info', methods=['GET'])
def model_info():
    return jsonify({
        'features': feature_names,
        'contract_types': list(le_contract.classes_),
        'payment_methods': list(le_payment.classes_)
    })


if __name__ == '__main__':
    print("\n" + "="*50)
    print("  Open: http://127.0.0.1:5000")
    print("="*50)
    app.run(debug=True, host='0.0.0.0', port=5000)