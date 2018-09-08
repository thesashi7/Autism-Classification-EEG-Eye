import numpy as np
from config.config import *
from utils.writer import *
from utils.analysis import splitSample

# need to combine train and test
def combine(eeg_file, eye_file):
    eeg_data = np.genfromtxt(eeg_file, delimiter=',', invalid_raise = False)
    eye_data = np.genfromtxt(eye_file, delimiter=',', invalid_raise = False)
    eye_data = np.delete(eye_data, eye_data.shape[1]-1, 1)
    combined_data = []
    i = 0
    while(i < len(eye_data)):
        j = 0
        while(j < len(eeg_data)):
            if(eye_data[i][0] == eeg_data[j][0]):
                cur_eeg = np.delete(eeg_data[j], 0, 0)
                comb = np.concatenate((eye_data[i],cur_eeg))
                combined_data.append(comb)
                break
            j+=1
        i+=1
    return combined_data

def createSamples(num_sam = 5, feature_type = "sh_entropy"):
    #divide this data into num_sam
    #return
    splitted = splitSample(main_path + "train-combined-"+feature_type+".csv",num_splits=num_sam)
    print len(splitted)
    for sam in splitted:
        print sam[1].shape
    return splitted

feature_type = "fft"
train_data = combine(main_path+"train-"+feature_type+"-eeg.csv", base_path+"train_eyetrack.csv")
test_data = combine(main_path+"test-"+feature_type+"-eeg.csv", base_path+"test_eyetrack.csv")



write_csv(np.array(train_data),main_path+"train-combined-"+feature_type+".csv")
write_csv(np.array(test_data),main_path+"test-combined-"+feature_type+".csv")
writeFeatures(main_path+"train-combined-"+feature_type+".csv","train-comb-"+feature_type)
writeFeatures(main_path+"test-combined-"+feature_type+".csv","test-comb-"+feature_type)