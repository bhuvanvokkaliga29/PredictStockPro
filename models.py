from keras.models import load_model

def load_trained_model(model_path):
    return load_model(model_path)

def predict_stock_prices(model, x_test, scaler):
    y_pred = model.predict(x_test)
    scale_factor = 1 / scaler.scale_[0]
    y_pred = y_pred * scale_factor
    return y_pred
