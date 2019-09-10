from sklearn.metrics import average_precision_score
import numpy as np
import read_file as read
import csv
from collections import OrderedDict
import configparser
import os

config = configparser.ConfigParser()
config.read('resource/data.conf')

output_dir = config['ARC_DATA']['processed']
modes = ["train","dev","test"]
bert_input = config['ARC_DATA']['bert_input']
bert_output = config['ARC_DATA']['bert_output']
bert_output_distribution= config['ARC_DATA']['distribution_out']

def mean_ap(y_true,y_pred):
    ap = 0.0
    pre_1 = 0
    for i in range(len(y_true)):
        ap += average_precision_score(y_true[i],y_pred[i])
        if y_true[i][np.argmax(y_pred[i])] ==1:
            pre_1+=1
    map = ap/(i+1)
    pre_1 = pre_1/(i+1)
    return map,pre_1

def generate_dist_file(question_ids,dist,label2int,file_name):
    file = open(file_name, "w+")
    output = []
    int2label = {j:i for i,j in label2int.items()}
    for idx, id in enumerate(question_ids):
        line = id + "\t"
        line_pred = {}
        pred_dist = dist[idx]
        for i in range(len(label2int)):
            line_pred[int2label[i]] = pred_dist[i]
        line_pred_ordered = OrderedDict(sorted(line_pred.items(),key = lambda t:t[1],reverse=True))
        for key, value in line_pred_ordered.items():
            line += key +":" + str(value) +"\t"
        line += "\n"
        output.append(line)
    file.writelines(output)
    file.close()

def generate_dist(mode):
    for layer in range(1,7):
        label2int = read.read_from_json(os.path.join(output_dir, str(layer) + "/label2int"))


        if mode == "dev":
            question_id = read.read_from_json(os.path.join(output_dir, "question_id_"+mode))
            dev_pred = read.read_from_tsv(os.path.join(bert_output, str(layer) + "/eval_results.tsv"))
            dev_pred = np.asarray(dev_pred, np.float32)
            generate_dist_file(question_id,dev_pred,label2int,os.path.join(bert_output_distribution,"dev/BERT-Base-DEV.L" + str(layer) +".classdist.txt"))

        elif mode == "test":
            question_id = read.read_from_json(os.path.join(output_dir, "question_id_"+mode))
            dev_pred = read.read_from_tsv(os.path.join(bert_output,  str(layer) + "/test_results.tsv"))
            dev_pred = np.asarray(dev_pred, np.float32)
            generate_dist_file(question_id,dev_pred,label2int,os.path.join(bert_output_distribution,"test/BERT-Base-TEST.L" + str(layer) +".classdist.txt"))

def multihot(label_q,label2int):
    multihot = np.zeros(len(label2int))
    for label in label_q:
        multihot[label2int[label]] = 1
    return list(multihot)

def evaluate(mode):

    for layer in range(1,7):
        label2int = read.read_from_json(os.path.join(output_dir, str(layer) + "/label2int"))
        if mode =="dev":
            dev_pred = read.read_from_tsv(os.path.join(bert_output, str(layer) + "/eval_results.tsv"))
            dev_pred =np.asarray(dev_pred, np.float32)
            labels = read.read_from_json(os.path.join( output_dir, str(layer) + "/label_" + mode))
            multihots = []
            for label_q in labels:
                multihots.append(multihot(label_q,label2int))
            dev_true = np.asarray(multihots)
            map, pre_1 = mean_ap(dev_true,dev_pred)
            print("layer "+ str(layer) , map,pre_1)
        elif mode =="test":
            dev_pred = read.read_from_tsv(os.path.join(bert_output,  str(layer) + "/test_results.tsv"))
            dev_pred =np.asarray(dev_pred, np.float32)
            labels = read.read_from_json(os.path.join( output_dir, str(layer) + "/label_" + mode))
            multihots = []
            for label_q in labels:
                multihots.append(multihot(label_q,label2int))
            dev_true = np.asarray(multihots)
            map, pre_1 = mean_ap(dev_true,dev_pred)
            print("layer "+ str(layer) , map,pre_1)

evaluate("dev")
evaluate("test")
generate_dist("dev")
generate_dist("test")