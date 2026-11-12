import pickle
with open('Iris_Model.pkl','rb') as file:
    model = pickle.load(file)

    sample = [[1.3, 3.4, 4.2,  3.4]]

    prediction = model.predict(sample)

    print("Prediction of the sample is:", prediction[0])