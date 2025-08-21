from omegaconf import DictConfig

from asr.lm.transformer_lm import TransformerForLanguageModel
from asr.models import ASRModel, register_model
from asr.models.transformer_lm.configurations  import TransformerLanguageModelConfigs
from asr.tokenizers.tokenizer import Tokenizer


@register_model("tranformer_lm", dataclass=TransformerLanguageModelConfigs)
class TransformerLanguageModel(ASRModel):
    def __init__(self, configs: DictConfig, tokenizer: Tokenizer) -> None:
        super(TransformerLanguageModel, self).__init__(configs, tokenizer)

        self.lm = TransformerForLanguageModel(
            num_classes=self.num_classes,
            max_length=self.configs.model.max_length,
            d_model=self.configs.model.d_model,
            d_ff=self.configs.model.d_ff,
            num_attention_heads=self.configs.model.num_attention_heads,
            pad_id=self.tokenizer.pad_id,
            sos_id=self.tokenizer.sos_id,
            eos_id=self.tokenizer.eos_id,
            dropout_p=self.configs.model.dropout_p,
            num_layers=self.configs.model.num_layers,
        )
