import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ================================
# Train model in background
# ================================
@st.cache_resource
def train_model():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']]
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
    
    X = df[['Pclass', 'Sex', 'Age', 'Fare']]
    y = df['Survived']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    return model, scaler

# ================================
# App UI
# ================================
st.title("🚢 Titanic Survival Predictor")
st.write("Enter passenger details to predict survival!")

# Load model
model, scaler = train_model()

# Input fields
st.subheader("Passenger Details")

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class", [1, 2, 3],
                          help="1=First, 2=Second, 3=Third")
    age = st.slider("Age", 1, 80, 25)

with col2:
    sex = st.selectbox("Gender", ["Female", "Male"])
    fare = st.slider("Fare Paid ($)", 1, 500, 50)

# Convert gender
sex_num = 0 if sex == "Female" else 1

# Predict button
if st.button("🔮 Predict Survival", type="primary"):
    
    # Prepare input
    input_data = scaler.transform([[pclass, sex_num, age, fare]])
    
    # Predict
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    
    st.divider()
    
    if prediction == 1:
        st.success(f"✅ Survived! Survival probability: {probability[1]*100:.1f}%")
        st.balloons()
    else:
        st.error(f"❌ Did not survive. Survival probability: {probability[1]*100:.1f}%")
    
    # Show input summary
    st.subheader("Passenger Summary")
    st.write(f"Class: {pclass} | Gender: {sex} | Age: {age} | Fare: ${fare}")
