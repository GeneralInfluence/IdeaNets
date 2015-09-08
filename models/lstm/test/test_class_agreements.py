__author__ = 'dogfish'

import numpy as np
import matplotlib as ml
import matplotlib.pyplot as plt
from models.lstm.scode.lstm_class import LSTM as lstm

class_tests = range(2,20,1)
acc_tests = np.array(range(60,96,5))/float(100)
acc_tests = acc_tests.tolist()
LSTM = lstm()

res = np.zeros(shape=(len(acc_tests),len(class_tests)))

# row = np.zeros(len(class_tests)).tolist()
# res = []
# for r in range(len(acc_tests)):
#     res.append(row)

for c,ct in enumerate(class_tests):
    for r,at in enumerate(acc_tests):

        res[r][c] = LSTM.calc_accuracy_baseline(ct,agreement=at)

def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)

# fig = plt.figure(figsize=(6, 3.2))
#
# ax = fig.add_subplot(111)
# ax.set_title('Accuracy Baseline')
# plt.imshow(np.array(res))
# ax.set_aspect('equal')
#
# # cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
# # cax.get_xaxis().set_visible(False)
# # cax.get_yaxis().set_visible(False)
# # cax.patch.set_alpha(0)
# # cax.set_frame_on(False)
# plt.colorbar(orientation='vertical')
# plt.show()


fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('Choice vs Agreement')
plt.imshow(res, interpolation='none', extent=[class_tests[0],class_tests[-1],acc_tests[-1],acc_tests[0]])
ax.set_ylabel('Probability of Agreement')
ax.set_xlabel('Number of Classes')
ax.set_aspect(2)
ax.set_aspect('auto')
forceAspect(ax,aspect=1)
plt.colorbar(orientation='vertical').set_label('Probability of People Choosing the Correct Class')
fig.savefig('/Users/dogfish/GI/git/IdeaNets/models/lstm/test/class_agreement.png')
# fig.savefig('auto.png')
# fig.savefig('force.png')