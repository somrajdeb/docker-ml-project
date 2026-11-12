# ci/cd pipeline will run this test.py file to check the accuracy of the model
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
model = joblib.load("Iris_Model.pkl")
test_df = pd.read_csv("Iris - Iris.csv.xls")
test_df=test_df.drop("Id",axis=1)
x=test_df.drop("Species",axis=1)
y=test_df["Species"]
pred=model.predict(x)
acc=accuracy_score(pred,y)
print("Accuracy of the model is:",acc)

assert len(pred)==len(y)
assert acc>=0.5
# ci/cd pipeline will fail if the accuracy is less than 50%

print("All tests passed successfully!")

