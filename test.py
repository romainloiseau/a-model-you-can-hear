import torch
import logging

import hydra

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

import matplotlib.pyplot as plt

from amodelyoucanhear.model import amodelyoucanhear
plt.switch_backend('agg')

import amodelyoucanhear

logger = logging.getLogger(__name__)

@hydra.main(config_path="configs", config_name="defaults")
def main(cfg: DictConfig) -> None:
    if cfg.seed != 0: pl.seed_everything(cfg.seed)
    logger.info("\n" + OmegaConf.to_yaml(cfg))

    datamodule = hydra.utils.instantiate(cfg.model.data)
    datamodule.setup()

    print(datamodule)
    
    model = hydra.utils.instantiate(
        cfg.model,
        _recursive_=False,
    )

    if isinstance(model, amodelyoucanhear.model.AModelYouCanHear):
        cfg.model.n_init_proposals = 1
        model.__initprotos__(datamodule.generate_protos(cfg.model))

    trainer = amodelyoucanhear.trainers.get_trainer(cfg)

    trainer.test(model, datamodule=datamodule,
        ckpt_path=cfg.model.load_weights
    )
    
if __name__ == '__main__':
    main()