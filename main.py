import torch
import logging

import hydra

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

import matplotlib.pyplot as plt
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
        model.__initprotos__(datamodule.generate_protos(cfg.model))

    trainer = amodelyoucanhear.trainers.get_trainer(cfg)

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)
    
if __name__ == '__main__':
    main()