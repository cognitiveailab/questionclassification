import read_file as read
import configparser
import os
import random
from collections import Counter

config = configparser.ConfigParser()
config.read('resource/data.conf')
data_path= config['GRAD_DATA']['raw_input_train']
processed_dir = config['GRAD_DATA']['processed']
bert_input = config['GRAD_DATA']['bert_input']
modes = ["train","dev","test"]


def generate_tsv(texts, labels, label2int, file_path):
    dataset = []
    for idx in range(len(texts)):
        label = labels[idx]
        text = texts[idx]
        dataset.append([text, str(label2int[label])])
    read.save_in_tsv(file_path, dataset)

def k_fold_cross_validation(items, k, randomize=False):

    if randomize:
        items = list(items)
        random.seed(20191705)
        random.shuffle(items)

    slices = [items[i::k] for i in range(k)]
    train = [[0,1,2],[1,2,3],[2,3,4],[3,4,0],[4,0,1]]
    validation = [3,4,0,1,2]
    test = [4,0,1,2,3]
    for i in range(k):
        validation_slice = slices[validation[i]]
        test_slice = slices[test[i]]
        train_slice = [idx for fold in train[i] for idx in slices[fold]]
        yield  train_slice, validation_slice, test_slice

def textfile2list(path):
    data = read.read_from_dir(path)
    labels = []
    questions = []
    for line in data.splitlines():
        text = line.split("|")
        labels.append(text[0])
        questions.append(text[1])
    return labels, questions


def cross_validation():
    data = read.textfile2list(data_path)
    counter_label = dict(Counter(data[0]))
    label2int = {label:idx for idx,label in enumerate(counter_label.keys())}
    read.save_in_json(os.path.join(bert_input, "label2int"), label2int)
    questions, label = data
    data_set= list(zip(questions,label))
    question_idx = list(range(0, len(questions)))
    for fold, [train_idx, validation_idx,test_idx] in enumerate(k_fold_cross_validation(question_idx, 5, randomize=True)):
        train = [data_set[question_id] for question_id in train_idx]
        validation = [data_set[question_id] for question_id in validation_idx]
        test = [data_set[question_id] for question_id in test_idx]
        input_data = [train, validation, test]
        for idx, mode in enumerate(modes):
            input = [dataitem[1] for dataitem in input_data[idx]]
            label = [dataitem[0] for dataitem in input_data[idx]]
            read.save_in_json(os.path.join(processed_dir, "question_" + mode +"_"+ str(fold)), input_data[idx])
            generate_tsv(input, label, label2int,os.path.join(bert_input,"grad_" + mode +"_"+ str(fold) + ".tsv"))

def main():
    cross_validation()