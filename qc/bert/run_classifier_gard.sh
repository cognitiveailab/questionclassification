export BERT_BASE_DIR=bert/model/uncased_L-12_H-768_A-12
export GARD_DIR=resource/data/gard/bert

python3.6 run_classifier_qc.py \
--task_name=gard \
--folder=0 \
--do_train=true \
--do_eval=true \
--do_predict=true \
--data_dir=$GARD_DIR \
--vocab_file=$BERT_BASE_DIR/vocab.txt \
--bert_config_file=$BERT_BASE_DIR/bert_config.json \
--init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
--max_seq_length=256 \
--train_batch_size=16 \
--learning_rate=5e-5 \
--num_train_epochs=5 \
--output_dir=output/gard/0/

python3.6 run_classifier_qc.py \
--task_name=gard \
--folder=1 \
--do_train=true \
--do_eval=true \
--do_predict=true \
--data_dir=$GARD_DIR \
--vocab_file=$BERT_BASE_DIR/vocab.txt \
--bert_config_file=$BERT_BASE_DIR/bert_config.json \
--init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
--max_seq_length=256 \
--train_batch_size=16 \
--learning_rate=5e-5 \
--num_train_epochs=5 \
--output_dir=output/gard/1/

python3.6 run_classifier_qc.py \
--task_name=gard \
--folder=2 \
--do_train=true \
--do_eval=true \
--do_predict=true \
--data_dir=$GARD_DIR \
--vocab_file=$BERT_BASE_DIR/vocab.txt \
--bert_config_file=$BERT_BASE_DIR/bert_config.json \
--init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
--max_seq_length=256 \
--train_batch_size=16 \
--learning_rate=5e-5 \
--num_train_epochs=5 \
--output_dir=output/gard/2/

python3.6 run_classifier_qc.py \
--task_name=gard \
--folder=3 \
--do_train=true \
--do_eval=true \
--do_predict=true \
--data_dir=$GARD_DIR \
--vocab_file=$BERT_BASE_DIR/vocab.txt \
--bert_config_file=$BERT_BASE_DIR/bert_config.json \
--init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
--max_seq_length=256 \
--train_batch_size=16 \
--learning_rate=5e-5 \
--num_train_epochs=5 \
--output_dir=output/gard/3/

python3.6 run_classifier_qc.py \
--task_name=gard \
--folder=4 \
--do_train=true \
--do_eval=true \
--do_predict=true \
--data_dir=$GARD_DIR \
--vocab_file=$BERT_BASE_DIR/vocab.txt \
--bert_config_file=$BERT_BASE_DIR/bert_config.json \
--init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
--max_seq_length=256 \
--train_batch_size=16 \
--learning_rate=5e-5 \
--num_train_epochs=5 \
--output_dir=output/gard/4/


