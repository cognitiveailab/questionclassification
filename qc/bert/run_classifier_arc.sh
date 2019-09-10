export BERT_BASE_DIR=bert/model/uncased_L-12_H-768_A-12
export ARC_DIR=resource/data/arc/bert

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
--output_dir=output/arc/1/

python3.6 run_classifier_qc.py \
--task_name=arc \
--folder=2 \
--do_train=true \
--do_eval=true \
--do_predict=true \
--data_dir=$ARC_DIR \
--vocab_file=$BERT_BASE_DIR/vocab.txt \
--bert_config_file=$BERT_BASE_DIR/bert_config.json \
--init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
--max_seq_length=256 \
--train_batch_size=16 \
--learning_rate=5e-5 \
--num_train_epochs=25 \
--output_dir=output/arc/2/

python3.6 run_classifier_qc.py \
--task_name=arc \
--folder=3 \
--do_train=true \
--do_eval=true \
--do_predict=true \
--data_dir=$ARC_DIR \
--vocab_file=$BERT_BASE_DIR/vocab.txt \
--bert_config_file=$BERT_BASE_DIR/bert_config.json \
--init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
--max_seq_length=256 \
--train_batch_size=16 \
--learning_rate=5e-5 \
--num_train_epochs=25 \
--output_dir=output/arc/3/

python3.6 run_classifier_qc.py \
--task_name=arc \
--folder=4 \
--do_train=true \
--do_eval=true \
--do_predict=true \
--data_dir=$ARC_DIR \
--vocab_file=$BERT_BASE_DIR/vocab.txt \
--bert_config_file=$BERT_BASE_DIR/bert_config.json \
--init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
--max_seq_length=256 \
--train_batch_size=16 \
--learning_rate=5e-5 \
--num_train_epochs=25 \
--output_dir=output/arc/4/

python3.6 run_classifier_qc.py \
--task_name=arc \
--folder=5 \
--do_train=true \
--do_eval=true \
--do_predict=true \
--data_dir=$ARC_DIR \
--vocab_file=$BERT_BASE_DIR/vocab.txt \
--bert_config_file=$BERT_BASE_DIR/bert_config.json \
--init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
--max_seq_length=256 \
--train_batch_size=16 \
--learning_rate=5e-5 \
--num_train_epochs=25 \
--output_dir=output/arc/5/

python3.6 run_classifier_qc.py \
--task_name=arc \
--folder=6 \
--do_train=true \
--do_eval=true \
--do_predict=true \
--data_dir=$ARC_DIR \
--vocab_file=$BERT_BASE_DIR/vocab.txt \
--bert_config_file=$BERT_BASE_DIR/bert_config.json \
--init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
--max_seq_length=256 \
--train_batch_size=16 \
--learning_rate=5e-5 \
--num_train_epochs=25 \
--output_dir=output/arc/6/