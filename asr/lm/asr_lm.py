import torch.nn as nn


class ASRLanguageModelBase(nn.Module):
    r"""Abstract base class for ASR language models.

    Provides shared utilities (parameter counting and dropout updating) for
    all language model implementations. Subclasses must implement
    :meth:`forward` and :meth:`forward_step`.

    Examples::

        >>> class MyLM(ASRLanguageModelBase):
        ...     def forward(self, inputs, input_lengths):
        ...         pass
        ...     def forward_step(self, inputs, input_lengths):
        ...         pass
    """
    def __init__(self):
        super(ASRLanguageModelBase, self).__init__()

    def count_parameters(self) -> int:
        return sum([p.numel() for p in self.parameters()])

    def update_dropout(self, dropout_p: float) -> None:
        for name, child in self.named_children():
            if isinstance(child, nn.Dropout):
                child.p = dropout_p

    def forward_step(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def forward(self, *args, **kwargs) -> None:
        raise NotImplementedError
