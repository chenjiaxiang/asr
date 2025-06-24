import glob
import os
import tarfile
from typing import Tuple, Dict

import pandas as pd

def load_label(filepath: str) -> Tuple[Dict, Dict]:
    char2id = dict()
    id2char = dict()

    ch_labels = pd.read_csv(filepath, encoding="utf-8")

    id_list = ch_labels["id"]
    char_list = ch_labels["char"]

    for (id_, char) in zip(id_list, char_list):
        char2id[char] = id_
        id2char[id_] = char
    return char2id, id2char


def read_transcript(dataset_path: str) -> Dict:
    """
    Returns:
        transcripts (dict): All the transcripts from AISHELL dataset. They are represented by {audio id: transcript}.
    """
    transcripts_dict = dict()

    with open(os.path.join(dataset_path, "transcript", "aishell_transcript_v0.8.txt")) as f:
        for line in f.readlines():
            tokens = line.split()
            audio_path = tokens[0]
            transcript = " ".join(tokens[1:])
            transcripts_dict[audio_path] = transcript
    return transcripts_dict


def sentence_to_target(sentence: str, char2id: Dict) -> str:
    target = str()

    for ch in sentence:
        try:
            target += str(char2id[ch]) + " "
        except KeyError:
            continue
        
    return target[:-1]


def get_key(audio_file: str) -> str:
    """Given an audio path, return its ID."""
    return os.path.basename(audio_file)[:-4]


def generate_character_labels(dataset_path: str, vocab_path: str) -> None:
    transcripts, label_list, label_freq = list(), list(), list()

    with open(os.path.join(dataset_path, "transcript", "aishell_transcript_v0.8.txt")) as f:
        for line in f.readlines():
            tokens = line.split(" ")
            transcript = " ".join(tokens[1:])
            transcripts.append(transcript)

        for transcript in transcripts:
            for ch in transcript:
                if ch not in label_list:
                    label_list.append(ch)
                    label_freq.append(1)
                else:
                    label_freq[label_list.index(ch)] += 1

        # sort together Using zip
        label_freq, label_list = zip(*sorted(zip(label_freq, label_list), reverse=True))
        label = {"id": [0, 1, 2, 3], "char": ["<pad>", "<sos>", "<eos>", "<blank>"], "freq": [0, 0, 0, 0]}

        for idx, (ch, freq) in enumerate(zip(label_list, label_freq)):
            label["id"].append(idx + 4)
            label["char"].append(ch)
            label["freq"].append(freq)

        # TODO maybe wrong: identical copy?
        label["id"] = label["id"]
        label["char"] = label["char"]
        label["freq"] = label["freq"]

        label_df = pd.DataFrame(label)
        label_df.to_csv(vocab_path, encoding="utf-8", index=False)


def generate_character_script(dataset_path: str, manifest_file_path: str, vocab_path: str) -> None:
    tarfiles = glob.glob(os.path.join(dataset_path, "wav", f"*.tar.gz"))

    char2id, id2char = load_label(vocab_path)
    transcripts_dict = read_transcript(dataset_path)

    for f in tarfiles:
        tar = tarfile.open(f, mode="r:gz")
        tar.extractall(os.path.join(dataset_path, "wav"))
        tar.close()
        os.remove(f)
    
    with open(manifest_file_path, "w") as f:
        for split in ("train", "dev", "test"):
            audio_paths = glob.glob(os.path.join(dataset_path, "wav", f"{split}/*/*.wav")) # TODO fix
            keys = [audio_path for audio_path in audio_paths if get_key(audio_path) in transcripts_dict]

            transcripts = [transcripts_dict[get_key(key)] for key in keys]
            labels = [sentence_to_target(transcript, char2id) for transcript in transcripts]

            for idx, audio_path in enumerate(audio_paths):
                audio_paths[idx] = audio_path.replace(f"{dataset_path}/", "")  # TODO maybe wrong
            
            for (audio_path, transcript, label) in zip(keys, transcripts, labels):
                f.write(f"{audio_path}\t{transcript}\t{label}\n")
        

