import read_file as read
import configparser
import os
import random
from collections import defaultdict

config = configparser.ConfigParser()
config.read('resource/data.conf')
train = config['TREC_DATA']['raw_input_train']
dev = config['TREC_DATA']['raw_input_dev']
test = config['TREC_DATA']['raw_input_test']
processed_dir = config['TREC_DATA']['processed']
bert_input = config['TREC_DATA']['bert_input']
modes = ["train","dev","test"]

def k_fold_cross_validation(items, k, randomize=False):
    if randomize:
        items = list(items)
        random.shuffle(items)
    slices = [items[i::k] for i in range(k)]
    # read.save_in_json("data/input_new/processed_input_cv/question_idx", slices)
    for i in range(k):
        validation = slices[i]
        training = [item
                    for s in slices if s is not validation
                    for item in s]
        yield training, validation

def process():
    dataset_all = read.read_from_tsv(train)[1:]
    dataset_test = read.read_from_tsv(test)[1:]

    random.seed(2009221)
    question_idx = list(range(0, len(dataset_all)))
    folds = []
    for training, validation in k_fold_cross_validation(question_idx, 10, randomize=True):
        folds.append([training, validation])

    dataset_train = [dataset_all[question_id] for question_id in folds[0][0]]
    dataset_dev = [dataset_all[question_id] for question_id in folds[0][1]]
    data  = [dataset_train,dataset_dev,dataset_test]

    for idx, mode in enumerate(modes):
        questions  = [dataitem[5] for dataitem in data[idx]]
        labels = [dataitem[4] for dataitem in data[idx]]
        qids = [dataitem[0] for dataitem in data[idx]]
        read.save_in_json(os.path.join(processed_dir, "label_" + mode), labels)
        read.save_in_json(os.path.join(processed_dir, "question_" + mode), questions)
        read.save_in_json(os.path.join(processed_dir, "question_id_" + mode), qids)

def get_label_at_level(label,level):
    label = label.strip()
    label_new = "_".join(label.split('_',level)[:level])
    return label_new

def split_label(level,mode):
    labels = read.read_from_json(os.path.join(processed_dir, "label_" + mode))
    labels_level = []
    for label in labels:
        label_level = get_label_at_level(label,level)
        labels_level.append(label_level)
    read.save_in_json(os.path.join(processed_dir,str(level)+"/"+"label_"+mode),labels_level)

def label_dict():
    for layer in range(1,3):
        label_counts = defaultdict(float)
        for mode in modes:
            split_label(layer,mode)
            labels = read.read_from_json(os.path.join(processed_dir,str(layer)+"/"+"label_"+mode))
            for label in labels:
                label = label.strip()
                label_counts[label] +=1.0
        label2int = {j:i for i,j in enumerate(label_counts)}
        read.save_in_json(os.path.join(processed_dir, str(layer) + "/label_counts"),label_counts)
        read.save_in_json(os.path.join(bert_input, str(layer) + "/label2int"),label2int)

def generate_tsv(texts,labels,label2int,file_path,mode):

    dataset = []
    for index,text in enumerate(texts):
        label=labels[index]
        dataset.append([text,str(label2int[label])])
    read.save_in_tsv(file_path,dataset)


def data2bert():
    for layer in range(1,3):
        label2int = read.read_from_json(os.path.join(bert_input, str(layer) + "/label2int"))
        for mode in modes:
            labels = read.read_from_json(os.path.join(processed_dir,str(layer)+"/"+"label_"+mode))
            question_only_texts = read.read_from_json(os.path.join(processed_dir, "question_" + mode))
            generate_tsv(question_only_texts,labels,label2int,os.path.join(bert_input, str(layer)+"/trec_"+mode+"_"+str(layer)+".tsv"),mode)

def main():
    process()
    label_dict()
    data2bert()

main()