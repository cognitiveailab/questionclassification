# qc

## Requirements
Python version: 3.6.5
Python Package:
sklear 0.20.0, numpy 1.15.3
tensorflow-gpu 1.12.0 # CPU Version of TensorFlow.
tensorflow-gpu 1.12.0  # GPU version of TensorFlow.

## Usages
* 'resource/data.conf' -configuration file for inputs and outputs.
* 'process_arc.py` - process ARC dataset and generate input for BERT training.
* 'process_trec.py` - process TREC dataset and generate input for BERT training.
* 'process_gard.py` - process GARD dataset and generate input for BERT training.
* 'process_lat.py` - process MLBioMedLAT dataset and generate input for BERT training.
* 'evaluate_arc.py` - generate evaluation score and output probability distribution file.

To processe the ARC dataset, please add the input and output file path in resource/data.conf, and run:
```
$ python process_arc
```

To evaluate the prediction of ARC dataset, :
```
$ python evaluate_arc.py
```

To train the BERT model for all datasets, please see the bert/README.md, one example of training BERT on ARC:
```shell
export BERT_BASE_DIR=bert/model/uncased_L-12_H-768_A-12
export ARC_DIR=resource/data/ARC/bert

python3.6 run_classifier_qc.py \
--task_name=arc \
--folder=1 \
--do_train=true \
--do_eval=true \
--do_predict=true \
--data_dir=$ARC_DIR \
--vocab_file=$BERT_BASE_DIR/vocab.txt \
--bert_config_file=$BERT_BASE_DIR/bert_config.json \
--init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
--max_seq_length=256 \
--train_batch_size=16 \
--learning_rate=2e-5 \
--num_train_epochs=5 \
--output_dir=resource/arc/bert_output/1/
```
