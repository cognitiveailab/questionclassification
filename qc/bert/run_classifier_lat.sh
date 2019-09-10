export BERT_BASE_DIR=bert/model/uncased_L-12_H-768_A-12
export LAT_DIR=resource/data/lat/bert

python3.6 run_classifier_qc.py \
--task_name=lat \
--folder=0 \
--do_train=true \
--do_eval=true \
--do_predict=true \
--data_dir=$LAT_DIR \
--vocab_file=$BERT_BASE_DIR/vocab.txt \
--bert_config_file=$BERT_BASE_DIR/bert_config.json \
--init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
--max_seq_length=256 \
--train_batch_size=16 \
--learning_rate=5e-5 \
--num_train_epochs=10 \
--output_dir=output/lat/0/

python3.6 run_classifier_qc.py \
--task_name=lat \
--folder=1 \
--do_train=true \
--do_eval=true \
--do_predict=true \
--data_dir=$LAT_DIR \
--vocab_file=$BERT_BASE_DIR/vocab.txt \
--bert_config_file=$BERT_BASE_DIR/bert_config.json \
--init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
--max_seq_length=256 \
--train_batch_size=16 \
--learning_rate=5e-5 \
--num_train_epochs=10 \
--output_dir=output/lat/1/

python3.6 run_classifier_qc.py \
--task_name=lat \
--folder=2 \
--do_train=true \
--do_eval=true \
--do_predict=true \
--data_dir=$LAT_DIR \
--vocab_file=$BERT_BASE_DIR/vocab.txt \
--bert_config_file=$BERT_BASE_DIR/bert_config.json \
--init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
--max_seq_length=256 \
--train_batch_size=16 \
--learning_rate=5e-5 \
--num_train_epochs=10 \
--output_dir=output/lat/2/

python3.6 run_classifier_qc.py \
--task_name=lat \
--folder=3 \
--do_train=true \
--do_eval=true \
--do_predict=true \
--data_dir=$LAT_DIR \
--vocab_file=$BERT_BASE_DIR/vocab.txt \
--bert_config_file=$BERT_BASE_DIR/bert_config.json \
--init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
--max_seq_length=256 \
--train_batch_size=16 \
--learning_rate=5e-5 \
--num_train_epochs=10 \
--output_dir=output/lat/3/

python3.6 run_classifier_qc.py \
--task_name=lat \
--folder=4 \
--do_train=true \
--do_eval=true \
--do_predict=true \
--data_dir=$LAT_DIR \
--vocab_file=$BERT_BASE_DIR/vocab.txt \
--bert_config_file=$BERT_BASE_DIR/bert_config.json \
--init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
--max_seq_length=256 \
--train_batch_size=16 \
--learning_rate=5e-5 \
--num_train_epochs=10 \
--output_dir=output/lat/4/

python3.6 run_classifier_qc.py \
--task_name=lat \
--folder=5 \
--do_train=true \
--do_eval=true \
--do_predict=true \
--data_dir=$LAT_DIR \
--vocab_file=$BERT_BASE_DIR/vocab.txt \
--bert_config_file=$BERT_BASE_DIR/bert_config.json \
--init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
--max_seq_length=256 \
--train_batch_size=16 \
--learning_rate=5e-5 \
--num_train_epochs=10 \
--output_dir=output/lat/5/

python3.6 run_classifier_qc.py \
--task_name=lat \
--folder=6 \
--do_train=true \
--do_eval=true \
--do_predict=true \
--data_dir=$LAT_DIR \
--vocab_file=$BERT_BASE_DIR/vocab.txt \
--bert_config_file=$BERT_BASE_DIR/bert_config.json \
--init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
--max_seq_length=256 \
--train_batch_size=16 \
--learning_rate=5e-5 \
--num_train_epochs=10 \
--output_dir=output/lat/6/

python3.6 run_classifier_qc.py \
--task_name=lat \
--folder=7 \
--do_train=true \
--do_eval=true \
--do_predict=true \
--data_dir=$LAT_DIR \
--vocab_file=$BERT_BASE_DIR/vocab.txt \
--bert_config_file=$BERT_BASE_DIR/bert_config.json \
--init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
--max_seq_length=256 \
--train_batch_size=16 \
--learning_rate=5e-5 \
--num_train_epochs=10 \
--output_dir=output/lat/7/

python3.6 run_classifier_qc.py \
--task_name=lat \
--folder=8 \
--do_train=true \
--do_eval=true \
--do_predict=true \
--data_dir=$LAT_DIR \
--vocab_file=$BERT_BASE_DIR/vocab.txt \
--bert_config_file=$BERT_BASE_DIR/bert_config.json \
--init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
--max_seq_length=256 \
--train_batch_size=16 \
--learning_rate=5e-5 \
--num_train_epochs=10 \
--output_dir=output/lat/8/

python3.6 run_classifier_qc.py \
--task_name=lat \
--folder=9 \
--do_train=true \
--do_eval=true \
--do_predict=true \
--data_dir=$LAT_DIR \
--vocab_file=$BERT_BASE_DIR/vocab.txt \
--bert_config_file=$BERT_BASE_DIR/bert_config.json \
--init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
--max_seq_length=256 \
--train_batch_size=16 \
--learning_rate=5e-5 \
--num_train_epochs=10 \
--output_dir=output/lat/9/