DATASET_PATH=/root/workspace/learn/asr/asr/data
MANIFEST_FILE_PATH=/root/workspace/learn/asr/asr/data/main.txt
export PYTHON_PATH=/root/workspace/learn/asr/asr:$PYTHON_PATH
python ./asr_cli/hydra_train.py \
    dataset=librispeech \
    dataset.dataset_download=True \
    dataset.dataset_path=$DATASET_PATH \
    dataset.manifest_file_path=$MANIFEST_FILE_PATH \
    tokenizer=libri_subword \
    tokenizer.vocab_path=/root/workspace/learn/asr/asr/data/vocab_data/LibriSpeech/ \
    model=conformer_lstm \
    audio=fbank \
    lr_scheduler=warmup_reduce_lr_on_plateau \
    trainer=gpu \
    criterion=cross_entropy \
    trainer.logger=tensorboard \
    trainer.batch_size=16