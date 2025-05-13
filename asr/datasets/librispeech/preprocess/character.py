from typing import Tuple, Dict
import logging

import pandas as pd

from asr.datasets.librispeech.preprocess.preprocess import collect_transcritps

logger = logging.getLogger(__name__)

def _generate_character_labels(labels_dest: str) -> None:
    logger.info("create_char_labels started...")
    
    special_tokens=  ["<pad>", "<sos>", "<eos>", "<blank>"]
    tokens = special_tokens + list(" ABCDEFGHIGKLMNOPQRSTUVWXYZ")

    # sort together Using zip
    label = {
        "id": [x for x in range(len(tokens))],
        "char": tokens,
    }
    
    label_df = pd.DataFrame(label)
    label_df.to_csv(labels_dest, encoding="utf-8", index=False)

def _load_label(filepath: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    char2id = dict()
    id2char = dict()

    ch_labels = pd.read_csv(filepath, encoding="utf-8")

    id_list = ch_labels["id"]
    char_list = ch_labels["char"]

    for (id_, char) in zip(id_list, char_list):
        char2id[char] = id_
        id2char[id_] = char

    return char2id, id2char

def sentence_to_target(sentence: str, char2id: Dict[str, str]) -> str:
    target = str()
    for ch in sentence:
        try:
            target += str(char2id[ch]) + " "
        except KeyError:
            continue
    return target[:-1]

def generate_manifest_files(dataset_path: str, manifest_file_path: str, vocab_path: str) -> None:
    _generate_character_labels(vocab_path)
    char2id, id2char = _load_label(vocab_path)

    transcripts_collection = collect_transcritps(dataset_path)

    with open(manifest_file_path, "w") as f:
        for idx, part in enumerate(["train-960", "dev-clean", "dev-other", "test-clean", "test-other"]):
            for transcript in transcripts_collection[idx]:
                audio_path, transcript = transcript.split("|")
                label = sentence_to_target(transcript, char2id)
                f.write(f"{audio_path}\t{transcript}\t{label}\n")