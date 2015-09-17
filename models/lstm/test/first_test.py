__author__ = 'ying'

import os, inspect, sys, numpy
import cPickle as pkl

this_dir = os.path.realpath( os.path.abspath( os.path.split( inspect.getfile( inspect.currentframe() ))[0]))
ideanet_dir = os.path.realpath(os.path.abspath(os.path.join(this_dir,"../../../..")))
# params_directory = os.path.realpath(os.path.abspath(os.path.join(this_dir,"../params")))
if this_dir not in sys.path: sys.path.insert(0, this_dir)
if ideanet_dir not in sys.path: sys.path.insert(0, ideanet_dir)

pkl_file = os.path.realpath(os.path.abspath(os.path.join(this_dir,"./lstm_model.npz")))

from IdeaNets.models.lstm.scode.lstm_class import LSTM as lstm

params={}
params["data_directory"] = "/home/ying/Deep_Learning/Synapsify_data"
Lpickle = lstm(params=params)
Lpickle.preprocess()
Lpickle.build_model()
Lpickle.train_model()
Lpickle.test_model()
'''
This part of code is to test if npz file object works or not
Lpickle = lstm()
Lpickle.build_model()
Lpickle.train_model()
Lpickle.test_model()



'''

'''
### My function
Lpickle.load_matrix(pkl_file)

print Lpickle.get_params("lstm_b")

### Using the numpy load function.
model_files = numpy.load(pkl_file)

params_list = ["lstm_b", "lstm_U", "lstm_W", "b", "U", "Wemb"]

for item in params_list:
    ### results from my function,
    print "paramter", item
    temp = Lpickle.get_params(item)

    ### compare with numpy load function
    print numpy.array_equal(model_files[item], temp)

test_sentence = ["The cow jumped over the moon."]
Lpickle.build_model()
pkl_res = Lpickle.classify(test_sentence)
print pkl_res
'''

'''
l= lstm()
l.build
train
test

l.pick_model()
'''


pkl_file = "/home/ying/Downloads/UnTrustworthy.pkl.eae7a621-4288-11e5-a82f-6c4008accfa6"
Lpickle=lstm()
Lpickle.load_pickle(pkl_file)
test_sentence = ["The cow jumped over the moon."]
pkl_res = Lpickle.classify(test_sentence)
print pkl_res

'''
Lpickle.ruofan_save(weight_file)

Lweight = lstm()
Lweight.ruofan_load(weight_file)
weight_res = Lweight.classify(test_sentence)

if pkl_res==weight_res:
    print "Test Passed!!"
'''