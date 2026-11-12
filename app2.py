import pickle
import joblib
form flask import Flask, request, jsonify
form prometheus_flask exporter import PrometheusMetrics

app=Flask(__name__)
app=prometheusMetrics(app)
model=joblib.load("Iris_Model.pkl")


@app.route("/")
def home():
    return jsonify({"message": "app is running"})

@app.route("/health")
def health():
    return jsonify({"message": "app is healthy"})

@app.route("/predict")
def predict():
    data=request.get_json()
    s_l=data["sepal_length"]
    s_w=data["sepal_width"]
    p_l=data["petal_length"]
    p_w=data["petal_width"]
    pred=model.predict([[s_l,s_w,p_l,p_w]])
    return jsonify({"prediction": pred[0]})


if __name__=="__main__":
    app.run(host="0.0.0.0", port=5001)