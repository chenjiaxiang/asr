from typing import List

from torch import Tensor


class Tokenizer(object):
    r"""Abstract base class for ASR tokenizers.

    Provides the interface that all concrete tokenizer implementations must
    satisfy. Stores special token IDs and exposes :meth:`encode` /
    :meth:`decode` methods used throughout the codebase.

    Args:
        (none — subclasses accept their own constructor arguments)

    Examples::

        >>> class MyTokenizer(Tokenizer):
        ...     def decode(self, labels: Tensor) -> str:
        ...         return " ".join(str(int(l)) for l in labels)
        ...     def encode(self, labels: str) -> List[int]:
        ...         return [int(t) for t in labels.split()]
    """
    def __init__(self, *args, **kwargs) -> None:
        self.sos_id = None
        self.eos_id = None
        self.pad_id = None
        self.blank_id = None

    def decode(self, labels: Tensor) -> str:
        raise NotImplementedError

    def encode(self, labels: str) -> List[int]:
        raise NotImplementedError

    def __call__(self, sentence):
        return self.encode(sentence)
