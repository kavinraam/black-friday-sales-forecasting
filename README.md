# Black Friday Purchase Prediction using Machine Learning

This project focuses on predicting customer purchase amounts during Black Friday sales using machine learning techniques. Built with Python and deployed via Streamlit, the model enables real-time predictions based on user demographics and shopping behavior.

## Project Description

During Black Friday, customer transactions spike dramatically, offering a valuable opportunity for retailers to analyze and predict consumer spending patterns. This project leverages the **Black Friday dataset** provided by Kaggle to build an intelligent regression system that estimates the purchase amount of individual users.

We processed over 500,000 transaction records, which include features such as gender, age, city category, and product information. Using advanced **feature engineering**, we created new metrics like:

- **Customer Loyalty Score**
- **Product Popularity Index**
- **Total Products Bought**

Three regression models were implemented and compared:

- **Linear Regression**
- **Random Forest Regressor**
- **XGBoost Regressor**

Among these, **XGBoost** provided the best performance with the highest accuracy and lowest error, making it ideal for deployment.

The final solution is a **Streamlit web app** that allows users to input customer attributes and instantly receive purchase predictions. The app also displays model performance metrics like MAE, RMSE, and R² Score, helping users evaluate prediction quality.

## Key Highlights

- Real-time prediction of customer purchases
- Feature engineering to enhance model accuracy
- Visual analysis using histograms, boxplots, and pie charts
- Deployed using Streamlit for ease of access and interaction
- Best model: **XGBoost Regressor** (R² Score: 0.7405)

## Use Cases

- E-commerce platforms can use this tool to:
  - Personalize marketing strategies
  - Optimize inventory and stock levels
  - Forecast revenue during high-volume events
  - Understand customer behavior trends

## Tools & Technologies

- Python, Pandas, NumPy
- Scikit-learn, XGBoost
- Matplotlib, Seaborn
- Streamlit

---

This project demonstrates how machine learning can empower businesses with predictive intelligence, especially during events like Black Friday where understanding consumer behavior is critical.
