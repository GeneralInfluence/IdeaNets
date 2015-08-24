"""
This is designed to ensure the synapsify_preprocess.py produces the same output as
the imdb_preprocess.py.
"""

from models.lstm.scode.lstm_class import LSTM as lstm
LSTM = lstm()
LSTM.load_pickle('UnTrustworthy.pkl.eae7a621-4288-11e5-a82f-6c4008accfa6')
pred = LSTM.classify(['The cow jumped over the moon.'])

print pred
