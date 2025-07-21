import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")
    df.fillna(-1, inplace=True)

    
    df["Gender"] = df["Gender"].map({"M": 1, "F": 0})
    df["City_Category"] = df["City_Category"].map({"A": 0, "B": 1, "C": 2})
    age_mapping = {'0-17': 0, '18-25': 1, '26-35': 2, '36-45': 3, '46-50': 4, '51-55': 5, '55+': 6}
    df["Age"] = df["Age"].map(age_mapping)
    df["Stay_In_Current_City_Years"] = df["Stay_In_Current_City_Years"].replace("4+", 4).astype(int)

    
    df["Customer_Loyalty_Score"] = df.groupby("User_ID")["Purchase"].transform("sum") / df.groupby("User_ID")["Product_ID"].transform("nunique")
    df["Product_Popularity_Index"] = df.groupby("Product_ID")["Purchase"].transform("sum")
    df["Total_Products_Bought"] = df.groupby("User_ID")["Product_ID"].transform("count")

    
    df.drop(["User_ID", "Product_ID"], axis=1, inplace=True)

    return df

df = load_data()

X = df.drop("Purchase", axis=1)
y = df["Purchase"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


@st.cache_resource
def train_models():
    models = {}

    
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    models["Linear Regression"] = lr_model

    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    models["Random Forest"] = rf_model

    
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    xgb_model.fit(X_train, y_train)
    models["XGBoost"] = xgb_model

    return models

models = train_models()


st.title("Black Friday Purchase Prediction")
st.sidebar.header("User Input Features")

def user_input():
    gender = st.sidebar.radio("Gender", ["Male", "Female"])
    age = st.sidebar.selectbox("Age Group", ['0-17', '18-25', '26-35', '36-45', '46-50', '51-55', '55+'])
    occupation = st.sidebar.slider("Occupation (0-20)", 0, 20, 5)
    city = st.sidebar.radio("City Category", ["A", "B", "C"])
    stay_years = st.sidebar.selectbox("Stay in Current City (Years)", [0, 1, 2, 3, 4])
    marital_status = st.sidebar.radio("Marital Status", ["Single", "Married"])
    product_category_1 = st.sidebar.slider("Product Category 1", 1, 20, 5)
    product_category_2 = st.sidebar.slider("Product Category 2", 1, 20, 5)
    product_category_3 = st.sidebar.slider("Product Category 3", 1, 20, 5)
    loyalty_score = st.sidebar.slider("Customer Loyalty Score", float(df["Customer_Loyalty_Score"].min()), float(df["Customer_Loyalty_Score"].max()))
    popularity_index = st.sidebar.slider("Product Popularity Index", float(df["Product_Popularity_Index"].min()), float(df["Product_Popularity_Index"].max()))
    total_products = st.sidebar.slider("Total Products Bought", int(df["Total_Products_Bought"].min()), int(df["Total_Products_Bought"].max()))

    
    gender = 1 if gender == "Male" else 0
    city = {"A": 0, "B": 1, "C": 2}[city]
    age = {'0-17': 0, '18-25': 1, '26-35': 2, '36-45': 3, '46-50': 4, '51-55': 5, '55+': 6}[age]
    marital_status = 1 if marital_status == "Married" else 0

    
    input_data = pd.DataFrame([[gender, age, occupation, city, stay_years, marital_status,
                                 product_category_1, product_category_2, product_category_3,
                                 loyalty_score, popularity_index, total_products]],
                                 columns=["Gender", "Age", "Occupation", "City_Category", "Stay_In_Current_City_Years",
                                       "Marital_Status", "Product_Category_1", "Product_Category_2", "Product_Category_3",
                                       "Customer_Loyalty_Score", "Product_Popularity_Index", "Total_Products_Bought"])
    return input_data


input_df = user_input()


st.sidebar.subheader("Choose a Model")
model_choice = st.sidebar.selectbox("Select Model", ["Linear Regression", "Random Forest", "XGBoost"])


if st.sidebar.button("Predict"):
    model = models[model_choice]
    prediction = model.predict(input_df)[0]
    st.subheader(f"Predicted Purchase Amount: ${prediction:.2f}")


st.subheader("Model Performance on Test Data")

@st.cache_data
def evaluate_models():
    performance = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        r2 = r2_score(y_test, y_pred)
        performance[name] = [mae, rmse, r2]
    
    return pd.DataFrame(performance, index=["MAE", "RMSE", "R² Score"])

performance_df = evaluate_models()
st.write(performance_df)

# Visualization of Purchase Distribution
st.subheader("Purchase Amount Distribution")
fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(df["Purchase"], bins=50, kde=True, color="blue", ax=ax)
ax.set_title("Distribution of Purchase Amounts")
st.pyplot(fig)
