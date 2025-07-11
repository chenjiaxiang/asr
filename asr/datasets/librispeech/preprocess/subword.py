from typing import List
import os
import shutil

import sentencepiece as spm

from asr.datasets.librispeech.preprocess.preprocess import collect_transcritps

SENTENCEPIECE_MODEL_NAME = "sp"

def _prepare_tokenizer(train_transcripts: List[str], vocab_size: int):
    """Prepare sentencepiece tokenizer"""
    input_file = "spm_input.txt"
    model_type = "unigram"

    with open(input_file, "w") as f:
        for transcript in train_transcripts:
            f.write(f"{transcript.split('|')[-1]}\n")

    spm.SentencePieceTrainer.Train(
        f"--input={input_file} "
        f"--model_prefix={SENTENCEPIECE_MODEL_NAME} "
        f"--vocab_size={vocab_size} "
        f"--model_type={model_type} "
        f"--pad_id=0 "
        f"--bos_id=1 "
        f"--eos_id=2 "
        f"--unk_id=3 "
        f"--user_defined_symbols=<blank>"
    )

def generate_manifest_files(dataset_path: str, manifest_file_path: str, vocab_path: str, vocab_size: int) -> None:
    transcripts_collection = collect_transcritps(dataset_path)
    _prepare_tokenizer(transcripts_collection[0], vocab_size)

    shutil.copy(f"{SENTENCEPIECE_MODEL_NAME}.model", os.path.join(vocab_path, f"{SENTENCEPIECE_MODEL_NAME}.model"))
    shutil.copy(f"{SENTENCEPIECE_MODEL_NAME}.vocab", os.path.join(vocab_path, f"{SENTENCEPIECE_MODEL_NAME}.vocab"))

    sp = spm.SentencePieceProcessor()
    sp.load(os.path.join(vocab_path, f"{SENTENCEPIECE_MODEL_NAME}.model"))

    with open(manifest_file_path, "w") as f:
        for idx, part in enumerate(["train-960", "dev-clean", "dev-other", "test-clean", "test-other"]):
            for transcript in transcripts_collection[idx]:
                audio_path, transcript = transcript.split("|")
                text = " ".join(sp.EncodeAsPieces(transcript))
                label = " ".join([str(item) for item in sp.EncodeAsIds(transcript)])
                f.write(f"{audio_path}\t{text}\t{label}\n")