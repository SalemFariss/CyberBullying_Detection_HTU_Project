import pickle

# SGDClassifier
def model(x):
    # load the model and predict the data.
    loaded_model = pickle.load(open('classification.model', 'rb'))

    y_predict = loaded_model.predict(x)

    return y_predict
