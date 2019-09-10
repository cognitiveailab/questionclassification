import read_file as read
import configparser
import os
import random
from collections import Counter

config = configparser.ConfigParser()
config.read('resource/data.conf')
data_path= config['LAT_DATA']['raw_input_train']
processed_dir = config['LAT_DATA']['processed']
bert_input = config['LAT_DATA']['bert_input']
modes = ["train","dev","test"]

def k_fold_cross_validation(items, k, randomize=False):

    if randomize:
        items = list(items)
        random.seed(20191705)
        random.shuffle(items)

    slices = [items[i::k] for i in range(k)]
    # read.save_in_json("data/input_lat/processed_input_cv/question_idx", slices)
    train = [[0,1,2,3,4,5,6,7],[1,2,3,4,5,6,7,8],[2,3,4,5,6,7,8,9],[3,4,5,6,7,8,9,0],[4,5,6,7,8,9,0,1],[5,6,7,8,9,0,1,2],[6,7,8,9,0,1,2,3],[7,8,9,0,1,2,3,4],[8,9,0,1,2,3,4,5],[9,0,1,2,3,4,5,6]]
    validation = [8,9,0,1,2,3,4,5,6,7]
    test = [9,0,1,2,3,4,5,6,7,8]
    for i in range(k):
        validation_slice = slices[validation[i]]
        test_slice = slices[test[i]]
        train_slice = [idx for fold in train[i] for idx in slices[fold]]
        print(set(train_slice) & set(test_slice))
        print(set(train_slice) & set(validation_slice))
        print(set(test_slice) & set(validation_slice))
        yield  train_slice, validation_slice, test_slice

def read_questions(path):

    data = read.read_from_tsv(path)[1:]

    labels = []
    questions = []
    labels_powerset = []
    labels_all = []
    for line in data:
        labels.append(line[3].split(","))
        labels_all+=line[3].split(",")
        labels_powerset.append(line[3])
        questions.append(line[2])

    counter_labels = dict(Counter(labels_all))
    label2int = {label:idx for idx,label in enumerate(counter_labels.keys())}
    read.save_in_json(os.path.join(bert_input, "label2int"), label2int)
    return questions, labels

def generate_tsv(texts,labels,label2int,file_path,mode):
    if mode == "Train" or mode.lower() == "train":
        dataset = []
        for index,text in enumerate(texts):
            for label in labels[index]:
                dataset.append([text,str(label2int[label])])
        read.save_in_tsv(file_path,dataset)
    else:
        dataset = []
        for index,text in enumerate(texts):
            label = labels[index]
            dataset.append([text,str(label2int[label[0]])])
        read.save_in_tsv(file_path,dataset)


def generate_cv():
    data = read_questions(data_path)
    label2int = read.read_from_json(os.path.join(bert_input,"label2int"))

    questions, label = data
    data_set= list(zip(questions,label))
    question_idx = list(range(0, len(questions)))
    for fold, [train_idx, validation_idx,test_idx] in enumerate(k_fold_cross_validation(question_idx, 10, randomize=True)):
        train = [data_set[question_id] for question_id in train_idx]
        validation = [data_set[question_id] for question_id in validation_idx]
        test = [data_set[question_id] for question_id in test_idx]
        data = [train, validation, test]
        for idx, mode in enumerate(modes):
            input = [dataitem[0] for dataitem in data[idx]]
            labels = [dataitem[1] for dataitem in data[idx]]
            idx_label = [[label2int[label_single]  for label_single in label] for label in labels ]
            read.save_in_json(os.path.join(processed_dir,"index_label_" + mode +"_"+ str(fold)), idx_label)
            read.save_in_json(os.path.join(processed_dir,"question_" + mode +"_"+ str(fold)), data[idx])
            generate_tsv(input, label, label2int,os.path.join(bert_input,"lat_" + mode +"_"+ str(fold) + ".tsv"),mode)

def main():
    generate_cv()

main()