import torch.nn as nn

class ASRDecoder(nn.Module):
    def __init__(self) -> None:
        super(ASRDecoder, self).__init__()

    def count_parameters(self) -> int:
        return sum([p.numel for p in self.parameters()])

    def update_dropout(self, dropout_p: float) -> None:
        for name, child in self.named_children():
            if isinstance(child, nn.Dropout):
                child.p = dropout_p

    def forward(self, *args, **kwargs):
        raise NotImplementedError