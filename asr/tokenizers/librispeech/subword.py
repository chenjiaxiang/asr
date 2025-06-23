import os
from dataclasses import dataclass, field

from omegaconf import DictConfig

from asr.dataclass.configurations import TokenizerConfigs
from asr.datasets.librispeech.preprocess.subword import SENTENCEPIECE_MODEL_NAME
from asr.tokenizers import register_tokenizer
from asr.tokenizers.tokenizer import Tokenizer
from asr.utils import SENTENCEPIECE_IMPORT_ERROR


@dataclass
class LibriSpeechSubwordTokenizerConfigs(TokenizerConfigs):
    unit: str = field(default="libri_subword", metadata={"help": "Unit of vocabulary."})
    sos_token: str = field(default="<s>", metadata={"help": "Start of sentence token."})
    eos_token: str = field(default="</s>", metadata={"help": "End of sentence token."})
    vocab_size: int = field(default=5000, metadata={"help": "Size of vocabulary."})
    vocab_path: str = field(default="../../../LibriSpeech/", metadata={"help": "Path of vocabulary file."})

@register_tokenizer("libri_subword", dataclass=LibriSpeechSubwordTokenizerConfigs)
class LibriSpeechSubwordTokenizer(Tokenizer):
    def __init__(self, configs: DictConfig):
        super(LibriSpeechSubwordTokenizer, self).__init__()
        try:
            import sentencepiece as spm
        except ImportError:
            raise ImportError(SENTENCEPIECE_IMPORT_ERROR)

        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(os.path.join(configs.tokenizer.vocab_path, f"{SENTENCEPIECE_MODEL_NAME}.model"))
        self.pad_id = self.sp.PieceToId(configs.tokenizer.pad_token)
        self.sos_id = self.sp.PieceToId(configs.tokenizer.sos_token)
        self.eos_id = self.sp.PieceToId(configs.tokenizer.eos_token)
        self.blank_id = self.sp.PieceToId(configs.tokenizer.blank_token)
        self.vocab_size = configs.tokenizer.vocab_size

    def __len__(self) -> int:
        return self.vocab_size
    
    def decode(self, labels):
        if len(labels.shape) == 1:
            return self.sp.DecodeIds([l.item() for l in labels])

        elif len(labels.shape) == 2:
            sentences = list()

            for label in labels:
                sentence = self.sp.DecodeIds([l.item() for l in label])
                sentences.append(sentence)

            return sentences

        else:
            raise ValueError("Unsupported label's shape")

    
    def encode(self, sentence: str) -> str:
        text = " ".join(self.sp.EncodeAsPieces(sentence))
        label = " ".join([str(self.sp.PieceToId(token)) for token in text])
        return label