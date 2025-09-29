import pickle
import streamlit as st
import pandas as pd


st.set_page_config("Insurance Price",page_icon="🧱")
st.write("Insurance Charges Prediction")

with open('insurance_model.pkl', 'rb') as file:
    pipeline = pickle.load(file)

df = pd.read_csv("insurance.csv")

age = st.selectbox("Age",options=range(1,101))
sex = st.selectbox("Sex",df['sex'].value_counts().index.to_list())
bmi = st.selectbox("BMI",df['bmi'].tolist())
children = st.select_slider("Childrens",options=range(0,21))
smoker = st.selectbox("Smoker",df['smoker'].value_counts().index.to_list())
region = st.selectbox("Region",df['region'].value_counts().index.to_list())


inputs = {
'age':age, 
'sex':sex, 
'bmi':bmi, 
'children':children, 
'smoker':smoker, 
'region':region
}
user_inputs = pd.DataFrame([inputs])

if st.button("Predict"):
    prediction = pipeline.predict(user_inputs)[0]
    st.success(f"Total Charges of your insurace will be ${prediction:.2f}")