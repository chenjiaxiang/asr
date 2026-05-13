import logging
from typing import Dict, Tuple

import pandas as pd

from asr.datasets.timit.preprocess.preprocess import collect_transcripts

logger = logging.getLogger(__name__)

# Characters shared with LibriSpeech character tokenizer so that a
# pre-trained vocab file can be reused across both datasets.
_SPECIAL_TOKENS = ["<pad>", "<sos>", "<eos>", "<blank>"]
_ALPHABET = list(" ABCDEFGHIJKLMNOPQRSTUVWXYZ'")


def _generate_character_labels(vocab_path: str) -> None:
    """Write a character-level vocabulary CSV file.

    The vocabulary is identical to the LibriSpeech character vocabulary so
    that a single tokenizer can be shared between both datasets.

    Args:
        vocab_path: Destination path for the CSV file.  The file has two
            columns: ``id`` (integer index) and ``char`` (the token).
    """
    tokens = _SPECIAL_TOKENS + _ALPHABET
    label_df = pd.DataFrame({"id": range(len(tokens)), "char": tokens})
    label_df.to_csv(vocab_path, encoding="utf-8", index=False)
    logger.info(f"Wrote {len(tokens)}-token vocabulary to {vocab_path}")


def _load_label(vocab_path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Load a character vocabulary CSV into bidirectional lookup dicts.

    Args:
        vocab_path: Path to a CSV file with ``id`` and ``char`` columns.

    Returns:
        A tuple ``(char2id, id2char)`` of lookup dictionaries.
    """
    char2id: Dict[str, int] = {}
    id2char: Dict[int, str] = {}

    ch_labels = pd.read_csv(vocab_path, encoding="utf-8")
    for id_, char in zip(ch_labels["id"], ch_labels["char"]):
        char2id[char] = int(id_)
        id2char[int(id_)] = char

    return char2id, id2char


def _sentence_to_target(sentence: str, char2id: Dict[str, int]) -> str:
    """Convert a transcript string to a space-separated sequence of token IDs.

    Characters not present in the vocabulary are silently skipped.

    Args:
        sentence: Upper-cased transcript string.
        char2id: Mapping from character to integer ID.

    Returns:
        Space-separated integer IDs as a string (e.g. ``"5 1 18 ..."``)
    """
    ids = [str(char2id[ch]) for ch in sentence if ch in char2id]
    return " ".join(ids)


def generate_manifest_files(
    dataset_path: str,
    manifest_file_path: str,
    vocab_path: str,
) -> None:
    """Generate a tab-separated manifest file for the TIMIT corpus.

    Each line of the manifest has three tab-separated fields::

        <audio_path>\\t<transcript>\\t<label_ids>

    where ``<audio_path>`` is relative to *dataset_path*, ``<transcript>``
    is the upper-cased text, and ``<label_ids>`` is the space-separated
    sequence of character IDs.

    The entries are written in split order: all training utterances first,
    then validation, then test, matching the convention used by the
    LibriSpeech data module.

    Args:
        dataset_path: Root directory of the TIMIT corpus.
        manifest_file_path: Destination path for the manifest ``.txt`` file.
        vocab_path: Path where the vocabulary CSV will be written (or
            overwritten if it already exists).
    """
    _generate_character_labels(vocab_path)
    char2id, _ = _load_label(vocab_path)

    logger.info("Collecting TIMIT transcripts (converting SPHERE WAV files if needed)…")
    audio_paths, transcripts, splits = collect_transcripts(dataset_path)

    with open(manifest_file_path, "w") as f:
        for audio_path, transcript, _split in zip(audio_paths, transcripts, splits):
            label = _sentence_to_target(transcript, char2id)
            f.write(f"{audio_path}\t{transcript}\t{label}\n")

    counts = {s: splits.count(s) for s in ("train", "valid", "test")}
    logger.info(
        f"Manifest written to {manifest_file_path} "
        f"(train={counts['train']}, valid={counts['valid']}, test={counts['test']})"
    )
