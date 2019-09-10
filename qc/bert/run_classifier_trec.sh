export BERT_BASE_DIR=bert/model/uncased_L-12_H-768_A-12
export TREC_DIR=resource/data/trec/bert

python3.6 run_classifier_qc.py \
--task_name=trec \
--folder=1 \
--do_train=true \
--do_eval=true \
--do_predict=true \
--data_dir=$TREC_DIR \
--vocab_file=$BERT_BASE_DIR/vocab.txt \
--bert_config_file=$BERT_BASE_DIR/bert_config.json \
--init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
--max_seq_length=128 \
--train_batch_size=16 \
--learning_rate=5e-5 \
--num_train_epochs=5 \
--output_dir=output/trec/coarse/

python3.6 run_classifier_qc.py \
--task_name=trec \
--folder=2 \
--do_train=true \
--do_eval=true \
--do_predict=true \
--data_dir=$TREC_DIR \
--vocab_file=$BERT_BASE_DIR/vocab.txt \
--bert_config_file=$BERT_BASE_DIR/bert_config.json \
--init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
--max_seq_length=128 \
--train_batch_size=16 \
--learning_rate=3e-5 \
--num_train_epochs=26 \
--output_dir=output/trec/fine/