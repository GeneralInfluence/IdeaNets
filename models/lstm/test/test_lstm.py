"""
This is designed to ensure the synapsify_preprocess.py produces the same output as
the imdb_preprocess.py.
"""

from models.lstm.scode.lstm_class import LSTM as lstm
LSTM = lstm()
LSTM.calc_accuracy_baseline(3)
# LSTM.load_pickle('UnTrustworthy.pkl.eae7a621-4288-11e5-a82f-6c4008accfa6')
LSTM.load_pickle('Naytevshares.pkl.45948945-4bda-11e5-aca5-6c4008accfa6')

# Test spelling
LSTM._model_options['correct_spelling'] = True
pred = LSTM.classify(['The cow jumped over the moon.','I wanna fuck that political blonde sitting across from me.'])
LSTM._model_options['correct_spelling'] = False
pred = LSTM.classify(['The cow jumped over the moon.'])

print pred
