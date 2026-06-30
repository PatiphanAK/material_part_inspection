"""Thin Hydra entrypoint for the part-inspection experiment.

main.py contains NO training or evaluation logic. It only composes the Hydra
config and dispatches on `cfg.mode.name` to the relevant module:

    mode=train    -> src.training.trainer.run_training
    mode=evaluate -> src.evaluate.run_evaluation

Example:
    python main.py mode=train env=local_rtx2080 model=resnet50 training=default data=default
    python main.py mode=evaluate env=local_rtx2080 model=resnet50
"""

import hydra
from omegaconf import DictConfig

from src.utils.logger import get_logger

log = get_logger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    mode = cfg.mode.name
    log.info("Dispatching mode=%s env=%s model=%s", mode, cfg.env.name, cfg.model.name)

    if mode == "train":
        from src.training.trainer import run_training

        run_training(cfg)
    elif mode == "evaluate":
        from src.evaluate import run_evaluation

        run_evaluation(cfg)
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Expected 'train' or 'evaluate'.")


if __name__ == "__main__":
    main()
