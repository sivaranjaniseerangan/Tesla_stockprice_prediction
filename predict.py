import pickle


def stock_prediction(features):
    pickled_model = pickle.load(open('tesla_stock_prediction.pkl', 'rb'))
    stock_price = str(round(list(pickled_model.predict([features]))[0]))

    return str("stock price may be " + stock_price)
test_features=[-0.8066389441603816,
 -0.7851190876008608,
 -0.8025207584189699,
 0.022343601298884685]
stock_prediction(test_features)