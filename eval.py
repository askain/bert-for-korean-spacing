import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from typing import Callable, List, Tuple
from preprocessor import Preprocessor
from dataset import CorpusDataset, load_data
from net import SpacingBertModel

def get_dataloader(
    data_path: str, transform: Callable[[List, List], Tuple], batch_size: int
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
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader


if __name__ == "__main__":
    config = OmegaConf.load("config/eval_config.yaml")
    
    preprocessor = Preprocessor(config.max_len)
    test_dataloader = get_dataloader(
        config.test_data_path, preprocessor.get_input_features, config.eval_batch_size
    )
    model = SpacingBertModel(config, None, None, test_dataloader)
    checkpoint = torch.load(config.ckpt_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["state_dict"])

    trainer = pl.Trainer()
    res = trainer.test(model)