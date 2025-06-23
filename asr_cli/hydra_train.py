# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass
import os
import hydra
import sentencepiece
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_info

import sys
sys.path.append("/root/workspace/learn/asr/asr")
# pause here TODO

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

    # """
    # Debug start
    # """
    # for inputs, targets, input_lengths, target_length in data_module.val_dataloader():
    #     print(targets)
    # """
    # Debug end
    # """

    model = MODEL_REGISTRY[configs.model.model_name](configs=configs, tokenizer=tokenizer)

    trainer = get_pl_trainer(configs, num_devices, logger)
    trainer.fit(model, data_module)
    trainer.test(model, data_module)
    

if __name__=="__main__":
    hydra_train_init()
    hydra_main()

# pause here