
from models.lstm.scode.lstm_class import LSTM as lstm
LSTM = lstm()
LSTM.preprocess()
LSTM.build_model()
LSTM.train_model()