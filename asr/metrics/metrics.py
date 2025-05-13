from typing import Tuple
from torch import Tensor

import Levenshtein as Lev


class ErrorRate(object):
    def __init__(self, tokenizer) -> None:
        self.total_dist = 0.0
        self.total_length = 0.0
        self.tokenizer = tokenizer

    def __call__(self, targets: Tensor, y_hats: Tensor) -> float:
        dist, length = self._get_distance(targets, y_hats)
        self.total_dist += dist
        self.total_length += length
        return self.total_dist / self.total_length

    def _get_distance(self, targets: Tensor, y_hats: Tensor) -> Tuple[float, int]:
        total_dist = 0
        total_length = 0

        for (target, y_hat) in zip(targets, y_hats):
            s1 = self.tokenizer.decode(target)
            s2 = self.tokenizer.decode(y_hat)

            dist, length = self.metric(s1, s2)

            total_dist += dist
            total_length += length

        return total_dist, total_length

    def metric(self, *args, **kwargs) -> Tuple[float, int]:
        raise NotImplementedError

    
class CharacterErrorRate(ErrorRate):
    def __init__(self, tokenizer):
        super(CharacterErrorRate, self).__init__(tokenizer)

    def metric(self, s1: str, s2: str) -> Tuple[float, int]:
        s1 = s1.replace(" ", "")
        s2 = s2.replace(" ", "")

        if "_" in s1:
            s1 = s1.replace("_", "")

        if "_" in s2:
            s2 = s2.replace("_", "")

        dist = Lev.distance(s2, s1)
        length = len(s1.replace(" ", ""))

        return dist, length

    
class WordErrorRate(ErrorRate):
    def __init__(self, tokenizer):
        super(WordErrorRate, self).__init__(tokenizer)

    def metric(self, s1: str, s2: str) -> Tuple[float, int]:
        b = set(s1.split() + s2.split())
        word2char = dict(zip(b, range(len(b))))

        # map the words to a char array (Levenshtein packages only accepts strings)
        w1 = [chr(word2char[w]) for w in s1.split()]
        w2 = [chr(word2char[w]) for w in s2.split()]

        dist = Lev.distance("".join(w1), "".join(w2))
        length = len(s1.split())

        return dist, length