import csv
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union

from omegaconf import DictConfig

from asr.dataclass.configurations import TokenizerConfigs
from asr.tokenizers import register_tokenizer
from asr.tokenizers.tokenizer import Tokenizer


@dataclass
class TIMITCharacterTokenizerConfigs(TokenizerConfigs):
    """Configuration for the TIMIT character-level tokenizer.

    The vocabulary is identical to the LibriSpeech character vocabulary
    (special tokens + upper-case Latin alphabet + space + apostrophe) so
    that checkpoints can be shared across both corpora.
    """

    eos_token: str = field(default="<eos>", metadata={"help": "End of sentence token."})
    unit: str = field(
        default="timit_character",
        metadata={"help": "Vocabulary unit identifier."},
    )
    vocab_path: str = field(
        default="timit_labels.csv",
        metadata={"help": "Path to the character vocabulary CSV file."},
    )


@register_tokenizer("timit_character", dataclass=TIMITCharacterTokenizerConfigs)
class TIMITCharacterTokenizer(Tokenizer):
    """Character-level tokenizer for the TIMIT corpus.

    Loads a vocabulary CSV (two columns: ``id``, ``char``) and provides
    :meth:`encode` / :meth:`decode` methods compatible with the rest of
    the ASR framework.

    Args:
        configs: Hydra / OmegaConf configuration object.  Reads keys
            under ``configs.tokenizer``: ``vocab_path``, ``encoding``,
            ``sos_token``, ``eos_token``, ``pad_token``, ``blank_token``.
    """

    def __init__(self, configs: DictConfig) -> None:
        super().__init__()
        self.vocab_dict, self.id_dict = self.load_vocab(
            vocab_path=configs.tokenizer.vocab_path,
            encoding=configs.tokenizer.encoding,
        )
        self.labels = self.vocab_dict.keys()
        self.sos_id = int(self.vocab_dict[configs.tokenizer.sos_token])
        self.eos_id = int(self.vocab_dict[configs.tokenizer.eos_token])
        self.pad_id = int(self.vocab_dict[configs.tokenizer.pad_token])
        self.blank_id = int(self.vocab_dict[configs.tokenizer.blank_token])
        self.vocab_path = configs.tokenizer.vocab_path

    def __len__(self) -> int:
        return len(self.labels)

    def decode(self, labels: Union[List[int], List[List[int]]]) -> Union[str, List[str]]:
        """Decode a sequence (or batch) of token IDs into a string.

        Args:
            labels: 1-D or 2-D array-like of integer token IDs.  Decoding
                stops at the first EOS token; blank tokens are skipped.

        Returns:
            A single transcript string for 1-D input, or a list of strings
            for 2-D (batched) input.
        """
        if len(labels.shape) == 1:
            sentence = ""
            for label in labels:
                if label.item() == self.eos_id:
                    break
                if label.item() == self.blank_id:
                    continue
                sentence += self.id_dict[label.item()]
            return sentence

        sentences = []
        for batch in labels:
            sentence = ""
            for label in batch:
                if label.item() == self.eos_id:
                    break
                if label.item() == self.blank_id:
                    continue
                sentence += self.id_dict[label.item()]
            sentences.append(sentence)
        return sentences

    def encode(self, sentence: str) -> str:
        """Encode a transcript string into a space-separated ID sequence.

        Characters not present in the vocabulary are silently skipped.

        Args:
            sentence: Upper-cased transcript string.

        Returns:
            Space-separated integer IDs as a string.
        """
        ids = []
        for ch in sentence:
            if ch in self.vocab_dict:
                ids.append(str(self.vocab_dict[ch]))
        return " ".join(ids)

    def load_vocab(
        self, vocab_path: str, encoding: str = "utf-8"
    ) -> Tuple[Dict[str, str], Dict[int, str]]:
        """Load a character vocabulary CSV into bidirectional lookup dicts.

        Args:
            vocab_path: Path to a CSV with ``id`` and ``char`` columns.
            encoding: File encoding (default ``"utf-8"``).

        Returns:
            A tuple ``(char2id, id2char)``.

        Raises:
            IOError: If *vocab_path* does not exist.
        """
        unit2id: Dict[str, str] = {}
        id2unit: Dict[int, str] = {}

        try:
            with open(vocab_path, encoding=encoding) as f:
                reader = csv.reader(f)
                next(reader)  # skip header
                for row in reader:
                    unit2id[row[1]] = row[0]
                    id2unit[int(row[0])] = row[1]
            return unit2id, id2unit
        except IOError:
            raise IOError(
                f"Character label file (CSV format) not found: {vocab_path}"
            )
