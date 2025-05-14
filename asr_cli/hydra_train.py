import os
import hydra
import sentencepiece
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_info

from asr.dataclass.initialize import hydra_train_init
from asr.datasets import DATA_MODULE_REGISTRY
from asr.models import MODEL_REGISTRY
from asr.tokenizers import TOKENIZER_REGISTRY
from asr.utils import get_pl_trainer, parse_configs

@hydra.main(config_path=os.path.join("..", "asr", "configs"), config_name="train")
def hydra_main(configs: DictConfig) -> None:
    rank_zero_info(OmegaConf.to_yaml(configs))
    pl.seed_everything(configs.trainer.seed)

    logger, num_devices = parse_configs(configs)

    data_module = DATA_MODULE_REGISTRY[configs.dataset.dataset](configs)
    data_module.prepare_data()
    tokenizer = TOKENIZER_REGISTRY[configs.tokenizer.unit](configs)
    
    data_module.setup()

    model = MODEL_REGISTRY[configs.model.model_name](configs=configs, tokenizer=tokenizer)

    trainer = get_pl_trainer(configs, num_devices, logger)
    trainer.fit(model, data_module)
    trainer.test(model, data_module)
    

if __name__=="__main__":
    hydra_train_init()
    hydra_main()

# pause here