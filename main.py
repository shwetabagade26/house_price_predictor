import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

data = pd.read_csv("train.csv")

def convert_sqft_to_num(value):
    try:
        if "-" in str(value):  
            vals = value.split("-")
            return (float(vals[0]) + float(vals[1])) / 2  
        elif any(c.isalpha() for c in str(value)):  
            return np.nan  
        return float(value)
    except:
        return np.nan  

data["total_sqft"] = data["total_sqft"].astype(str).apply(convert_sqft_to_num)
data.dropna(subset=["total_sqft"], inplace=True)

data["price_per_sqft"] = data["price"] / data["total_sqft"]

data["price"] = np.log(data["price"])

data = pd.get_dummies(data, columns=['site_location'], drop_first=True)

x = data.drop(["id", "availability", "price"], axis=1)
y = data["price"]

def remove_outliers(column):
    q1, q3 = column.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr
    return np.clip(column, lower_limit, upper_limit)

x["total_sqft"] = remove_outliers(x["total_sqft"])
y = remove_outliers(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=0)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=0)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print(f"R2 Score on Test Set: {r2:.4f}")

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(list(x.columns), open("features.pkl", "wb"))
