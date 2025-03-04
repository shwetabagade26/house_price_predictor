import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

data = pd.read_csv("train.csv")

@app.route("/")
def index():
    locations = sorted(data["site_location"].unique())
    return render_template("home.html", locations=locations)

@app.route("/predict", methods=["POST"])
def predict():
    location = request.form.get("location")
    area = float(request.form.get("area"))
    bhk = int(request.form.get("bhk"))

    input_data = pd.DataFrame([[area, bhk]], columns=["total_sqft", "size"])

    location_data = pd.get_dummies(pd.Series([location]), prefix="site_location")

    for col in features:
        if col not in input_data.columns:
            input_data[col] = location_data[col] if col in location_data else 0

    input_scaled = scaler.transform(input_data)

    pred = model.predict(input_scaled)[0] * 1e6 * 2.5
    return str(np.round(pred, 2))

if __name__ == "__main__":
    app.run(debug=True, port=5000)
