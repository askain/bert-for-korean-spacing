from typing import Callable, List, Tuple
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from preprocessor import Preprocessor
from dataset import CorpusDataset, load_data
from net import SpacingBertModel


def get_dataloader(
    data_path: str, transform: Callable[[List, List], Tuple], batch_size: int, num_workers: int
) -> DataLoader:
    """dataloader 생성

    Args:
        data_path: dataset 경로
        transform: input feature로 변환해주는 funciton
        batch_size: dataloader batch size

    Returns:
        dataloader
    """
    dataset = CorpusDataset(load_data(data_path), transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    return dataloader




if __name__ == "__main__":
    config = OmegaConf.load("config/train_config.yaml")
    preprocessor = Preprocessor(config.max_len)
    train_dataloader = get_dataloader(
        config.train_data_path, preprocessor.get_input_features, config.train_batch_size, num_workers=config.cpus
    )
    val_dataloader = get_dataloader(
        config.val_data_path, preprocessor.get_input_features, config.train_batch_size, num_workers=config.cpus
    )
    test_dataloader = get_dataloader(
        config.test_data_path, preprocessor.get_input_features, config.eval_batch_size, num_workers=config.cpus
    )

    bert_finetuner = SpacingBertModel(
        config, train_dataloader, val_dataloader, test_dataloader
    )

    logger = TensorBoardLogger(save_dir=config.log_path, version=1, name=config.task)

    checkpoint_callback = ModelCheckpoint(
        filepath="checkpoints/{epoch}_{val_loss:3f}",
        verbose=True,
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        prefix="",
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.001,
        patience=3,
        verbose=False,
        mode="min",
    )

    trainer = pl.Trainer(
        gpus=config.gpus,
        distributed_backend=config.distributed_backend,
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=early_stop_callback,
        logger=logger,
    )

    trainer.fit(bert_finetuner)
    trainer.test()

