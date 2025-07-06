import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

st.title("💳 Credit Card Fraud Detection")

# File Upload
uploaded_file = st.file_uploader("📂 Upload your creditcard.csv file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.write("📊 Preview of Dataset:", data.head())

    # Balance the dataset
    legit = data[data.Class == 0]
    fraud = data[data.Class == 1]
    legit_sample = legit.sample(n=len(fraud), random_state=2)
    data = pd.concat([legit_sample, fraud], axis=0)

    # Split features and label
    X = data.drop(columns="Class", axis=1)
    y = data["Class"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Accuracy
    train_acc = accuracy_score(model.predict(X_train), y_train)
    test_acc = accuracy_score(model.predict(X_test), y_test)

    st.success(f"✅ Model trained!\nTraining Accuracy: {train_acc:.2f}\nTest Accuracy: {test_acc:.2f}")

    # Manual prediction input
    st.subheader("🧠 Test a Transaction")
    st.markdown("Paste 30 comma-separated feature values from a transaction row (excluding Class).")

    input_text = st.text_area("🔢 Input features (V1, V2, ..., V28, Amount, Time):", height=100)

    if st.button("🚀 Predict"):
        try:
            input_list = np.array([float(x) for x in input_text.split(',')])
            if input_list.shape[0] != X.shape[1]:
                st.error(f"❌ Expected {X.shape[1]} features, but got {input_list.shape[0]}")
            else:
                prediction = model.predict(input_list.reshape(1, -1))
                result = "✅ Legitimate Transaction" if prediction[0] == 0 else "🚨 Fraudulent Transaction"
                st.success(result)
        except ValueError:
            st.error("❌ Invalid input. Please enter only numeric values separated by commas.")
else:
    st.warning("⚠️ Please upload your `creditcard.csv` file to begin.")
