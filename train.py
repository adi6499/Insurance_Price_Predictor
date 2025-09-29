import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

df = pd.read_csv("insurance.csv")
df.head()
#lets see if anything is missing 
df.isnull().sum()
#no
df.info()
df.describe()
num_features = ['age','bmi','children',]
cat_features = ['sex','smoker','region']
#split and train the data 
X = df.drop(columns=['charges'])
Y = df['charges']
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

#let apply standard scaler for numberical and one hot encode for category
from sklearn.preprocessing import StandardScaler,OneHotEncoder

num_transformer = Pipeline(steps=[
    ('ss',StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('ohe',OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ('num',num_transformer,num_features),
    ('cat',cat_transformer,cat_features)
])

pipeline = Pipeline([
    ('preprocessor',preprocessor),
    ('regression',LinearRegression())
])
pipeline.fit(X_train,Y_train)
df.head()
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

y_pred = pipeline.predict(X_test)
mse = mean_squared_error(Y_test, y_pred)
mae = mean_absolute_error(Y_test, y_pred)
r2 = r2_score(Y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.4f}")

inputs = {
'age':19, 
'sex':'female', 
'bmi':25, 
'children':2, 
'smoker':'yes', 
'region':'northwest'
}
user_inputs = pd.DataFrame([inputs])

import joblib

joblib.dump(pipeline,"insurance.pkl")
print("Insurance.pkl successfully created ✅")