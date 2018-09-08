import numpy as np
import pandas as pd

def oneHotDecode(label):
    decoded = []
    for v in label:
        if v[1] == 1:
            decoded.append(1)
        else:
            decoded.append(0)
    return np.array(decoded)

def confusionMatrix(actual, predicted, shape=2):
    if (shape == 2):
        actual = pd.Series(oneHotDecode(actual), name='Actual')
        predicted = pd.Series(oneHotDecode(np.round(np.clip(predicted, 0, 1))), name='Predicted')

    df_confusion = pd.crosstab(actual, predicted, rownames=['Actual'], colnames=['Predicted'], margins=True)
    print df_confusion

def splitSample(featureFile, num_splits = 5):
    data = np.genfromtxt(featureFile, delimiter=',', invalid_raise=False)
    data = np.delete(data, 0, 1)
    np.random.shuffle(data)
    samples = np.array_split(data,num_splits)
    splitted = []
    for sam in samples:
        print sam.shape
        target = np.empty([sam.shape[0], 1])
        i = 0
        while i < sam.shape[0]:
            if (sam[i][sam.shape[1] - 1] == 1):
                target[i] = 0
            else:
                target[i] = 1
            i += 1
        print target.shape
        print target
        sam = np.delete(sam, np.s_[sam.shape[1] - 1: sam.shape[1]], axis=1)

        target = target.reshape((target.shape[0], 1))
        splitted.append([sam,target])
    return splitted