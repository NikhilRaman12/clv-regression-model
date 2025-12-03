#import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
import pickle
#read the csv by data frame
df= pd.read_csv("E‑Commerce Customer Behavior.csv")
print(df.head())

#data inspection and EDA
print(df.shape)
print(df.info())
print(df.describe())
print(df.dtypes)
print(df.columns)

#Handling null values
print(df.isnull().sum())
print(df.duplicated().sum())
print(df.dropna())
print(df.drop_duplicates(inplace=True))

#feature engineering
#transcation level custmer life time value prediction
df["CLV_txn"] = (df["Unit_Price"] * df["Quantity"] - df["Discount_Amount"]) * df["Customer_Rating"]
#aggregate to customer level by using customer id
clv_df= df.groupby("Customer_ID")["CLV_txn"].sum().reset_index()
clv_df= clv_df.rename(columns={"CLV_txn": "CLV"})
print(clv_df)

#merge features_df and merge_df 
features_df = df.groupby("Customer_ID")[[
    "Unit_Price","Quantity","Discount_Amount","Total_Amount","Customer_Rating"
]].mean().reset_index()

final_df = pd.merge(clv_df, features_df, on="Customer_ID")

#lets check with Linear regression model

y = final_df["CLV"]
X = final_df.drop(columns=["Customer_ID","CLV"])
X_train, X_test, y_train, y_test= train_test_split(X,y, random_state= 42, test_size=0.2)
model=LinearRegression()
model.fit(X_train, y_train)
y_pred= model.predict(X_test)

#evaluation
mse= mean_squared_error(y_test, y_pred)
mae= mean_absolute_error(y_test, y_pred)
r2= r2_score(y_test, y_pred)
print(f"Mean Squared Error:{mse}")
print(f"Mean Absolute Error: {mae}")
print(f"R-squared:{r2}")

#lets check with random forest regression 
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

#evaluation
mse= mean_squared_error(y_test, y_pred)
mae= mean_absolute_error(y_test, y_pred)
r2= r2_score(y_test, y_pred)
print(f"Mean Squared Error:{mse}")
print(f"Mean Absolute Error: {mae}")
print(f"R-squared:{r2}")

#correlation analysis
plt.figure(figsize=(10,6))
corr= final_df.drop(columns=["Customer_ID"]).corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap= "coolwarm")
plt.title("Correlation Heatmap")
plt.show()

#Feature Importance Analysis
import matplotlib.pyplot as plt
import seaborn as sns

importances = rf.feature_importances_
feat_importances = pd.Series(importances, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.title("Top Feature Importances for CLV Prediction")
plt.show()

#cross validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(rf, X, y, cv=5, scoring="r2")
print("Cross-validated R² scores:", scores)
print("Mean R²:", scores.mean())

#save the models
joblib.dump("rf_clv_model.pkl")
joblib.dump("linear_clv_model.pkl")

#Our RandomForest regression achieved an R² of 0.993, reducing mean absolute error to ~69 CLV units.
# By aggregating transaction-level features and applying nonlinear modeling, we captured nearly all variance in customer lifetime value. 
#Feature importance analysis highlights the drivers of CLV, making the model both accurate and interpretable.













