# import numpy as np
# from numpy import genfromtxt
#
# train_labels = genfromtxt('Labels/train_answers.csv', delimiter=',',dtype=np.int32)     #since original csv file has only label of distinct writers, we need to add same labels for the same writers for 4 documents each
#
# count=0
# for a in range(0,train_labels.shape[0]):
#     for b in range(0,3):
#         train_labels=np.insert(train_labels,count,train_labels[count])
#     count+=4
# np.savetxt("Labels/labels.csv", train_labels, delimiter=",", fmt="%i")