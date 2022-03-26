"""Test lstm."""


from src.models.lstm_model import LSTMPredictionModel

from src.data import load_data
from src import config
from src.preprocessing.dataset import generate_dataset

def test_lstm():
    """Test LSTM."""
    # Load the data and model
    model = LSTMPredictionModel()

    x_train, y_train = generate_dataset(
        load_data(data_path=config.DATA_FOLDER, year=2020), shift=20
    )
    x_val, y_val = generate_dataset(
        load_data(data_path=config.DATA_FOLDER, year=2019), shift=20
    )
    x_test, y_test = generate_dataset(
        load_data(data_path=config.DATA_FOLDER, year=2018), shift=20
    )
    model.fit(
        X_train=x_train,
        y_train=y_train,
        X_val=x_val,
        y_val=y_val,
        epochs=1,
        batch_size=32,
        plot_history=False
    )
