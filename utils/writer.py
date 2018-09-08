import csv
import numpy as np
from sklearn.model_selection import train_test_split
from config.config import *
import xlsxwriter

def sanitize(data):
    where_are_NaNs = np.isnan(data)
    data[where_are_NaNs] = 0.0
    return data

def write_csv(data, file_name):
    data = sanitize(data)
    with open(file_name, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
        # print(type(features[0]))
        if (isinstance(data, np.ndarray)):
            for fv in data:
                print(fv)
                writer.writerow(fv)
        else:  # only one feature vector
            writer.writerow(data)

# iterate over each subject
"""
    @param data: feature set to be written in the csvfile
    @param fileName: Name or path of the csvfile to be created and written into
"""
def writeCSVFile(data, fileName):
    with open(fileName, 'wb') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_NONE, escapechar='\\')
        for row in data:
            writer.writerow(row)

def writeFeatures(featureFile, type_name):
    data = np.genfromtxt(featureFile, delimiter=',', invalid_raise=False)
    data = np.delete(data, 0, 1)
    np.random.shuffle(data)
    print data.shape
    target = np.empty([data.shape[0], 1])
    i = 0
    while i < data.shape[0]:
        if (data[i][data.shape[1] - 1] == 1):
            target[i] = 0
        else:
            target[i] = 1
        i += 1
    print target.shape
    print target
    data = np.delete(data, np.s_[data.shape[1] - 1: data.shape[1]], axis=1)

    target = target.reshape((target.shape[0], 1))
    write_csv(data, main_path +type_name + "-x.csv")
    write_csv(target, main_path + type_name + "-y.csv")

def writeTrainAndTestFile(featureFile, type_name):
    data = np.genfromtxt(featureFile, delimiter=',', invalid_raise = False)
    data = np.delete(data,0,1)
    np.random.shuffle(data)
    print data.shape
    target = np.empty([data.shape[0], 1])
    i = 0
    while i < data.shape[0]:
        if (data[i][data.shape[1] - 1] == 1):
            target[i] = 0
        else:
            target[i] = 1
        i += 1
    print target.shape
    print target
    data = np.delete(data, np.s_[data.shape[1] - 1: data.shape[1]], axis=1)
    train_data, test_data, train_target, test_target = train_test_split(
        data, (target[:, np.newaxis]), test_size=0.2, random_state=42)
    test_target = test_target.reshape((test_target.shape[0], 1))
    train_target = train_target.reshape((train_target.shape[0], 1))
    write_csv(train_data, main_path + "train-"+type_name+"-x.csv")
    write_csv(train_target, main_path + "train-"+type_name+"-y.csv")
    write_csv(test_data, main_path + "test-"+type_name+"-x.csv")
    write_csv(test_target, main_path + "test-"+type_name+"-y.csv")

# expects an arra for data
# each entry of data will be written on each row
#
def writeExcel(data, file_name):
    workbook = xlsxwriter.Workbook(base_path+file_name+'.xlsx')
    worksheet = workbook.add_worksheet()
    # Start from the first cell. Rows and columns are zero indexed.
    row = 0
    col = 0

    # Iterate over the data and write it out row by row.
    for en in (data):
        worksheet.write(row, col, en)
        worksheet.write(row, col, en)
        row += 1


    workbook.close()

def standerizeLabel(label_file):
    data = np.genfromtxt(label_file, delimiter=',', dtype=None)
    new_label = []
    for l in data:
        if l == 1:
            new_label.append([0])
        else:
            new_label.append([1])

    print new_label
    return new_label