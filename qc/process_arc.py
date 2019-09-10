from collections import defaultdict
import read_file as read
import configparser
import os

config = configparser.ConfigParser()
config.read('resource/data.conf')
train = config['ARC_DATA']['raw_input_train']
dev = config['ARC_DATA']['raw_input_dev']
test = config['ARC_DATA']['raw_input_test']
processed_dir = config['ARC_DATA']['processed']
bert_input = config['ARC_DATA']['bert_input']
data = {"train": train,"dev": dev, "test": test}
modes = ["train","dev","test"]


def tsv2list(data,mode):
    dataset = read.read_from_tsv(data[mode])[1:]
    input  = [[dataitem[19],dataitem[17].strip()] for dataitem in dataset]
    qid = [dataitem[0] for dataitem in dataset]
    answer_choice = [dataitem[3] for dataitem in dataset]
    return input,answer_choice, qid

def split_question_answer(q_text):
    if "(A) " in q_text:
        q_text_new = q_text.split("(A) ")
        if len(q_text_new) == 2:
            question = q_text_new[0]
            answer = "(A) " + q_text_new[1]
            return question, answer

    elif "(1) " in q_text:
        q_text_new = q_text.split("(1) ")
        if len(q_text_new) == 2:
            question = q_text_new[0]
            answer = "(1) " + q_text_new[1]
            return question, answer
    else:
        return None,None

def get_label_at_level(label_list,level):
    label_level = list()
    for label in label_list:
        label = label.strip()
        label_new = "_".join(label.split('_',level)[:level])
        label_level.append(label_new)
    return list(set(label_level))

def split_label(output_dir,labels, level,mode):
    labels_level = []
    for label in labels:
        label_list = label.split(", ")
        label_level = get_label_at_level(label_list,level)
        labels_level.append(label_level)
    read.save_in_json(os.path.join(output_dir,str(level)+"/label_"+mode),labels_level)

def process_inputs(output_dir,mode,inputs):

    print(mode, len(inputs))
    question_all_answers = [input[0] for input in inputs]
    labels = [input[1] for input in inputs]

    for level in range(1,7):
        split_label(output_dir,labels,level,mode)

    read.save_in_json(os.path.join(output_dir, "label_" + mode), labels)
    read.save_in_json(os.path.join(output_dir, "question+answer_"+mode),question_all_answers)

def label_dict(output_dir):
    for layer in range(1,7):
        label_counts = defaultdict(float)
        for mode in modes:
            labels = read.read_from_json(os.path.join(output_dir,  str(layer) + "/label_" + mode))
            for label_list in labels:
                for label in label_list:
                    label_counts[label] +=1.0
        read.save_in_json(os.path.join(output_dir, str(layer) + "/label_counts"),label_counts)
        label2int = {j: i for i, j in enumerate(label_counts)}
        read.save_in_json(os.path.join(bert_input, str(layer) + "/label2int"), label2int)

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


def data2bert():
    for layer in range(1,7):
        label2int = read.read_from_json(os.path.join(bert_input,str(layer) + "/label2int"))
        for mode in modes:
            labels = read.read_from_json( os.path.join(processed_dir, str(layer)+ "/label_"+ mode))
            question_answer_texts = read.read_from_json(os.path.join(processed_dir,"question+answer_"+mode))
            generate_tsv(question_answer_texts,labels,label2int,os.path.join(bert_input, str(layer)+"/arc_"+mode+"_"+str(layer)+".tsv"),mode)

def main():
    for mode in modes:
        inputs,answer_choices, qids = tsv2list(data,mode)
        read.save_in_json(os.path.join( processed_dir,"question_id_" + mode), qids)
        process_inputs(processed_dir,mode,inputs)
    label_dict(processed_dir)
    data2bert()

main()




