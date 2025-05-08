class Tokenizer(object):
    def __init__(self, *args, **kwargs) -> None:
        self.sos_id = None
        self.eos_id = None
        self.pad_id = None
        self.blank_id = None

    def decode(self, labels):
        raise NotImplementedError

    def encode(self, labels):
        raise NotImplementedError

    def __call__(self, sentence):
        return self.encode(sentence)