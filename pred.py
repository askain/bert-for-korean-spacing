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


config = OmegaConf.load("config/eval_config.yaml")
preprocessor = Preprocessor(config.max_len)
checkpoint = torch.load(config.ckpt_path, map_location=lambda storage, loc: storage)

model = SpacingBertModel(config, None, None, None)
model.load_state_dict(checkpoint["state_dict"])

trainer = pl.Trainer()

def spacing(text):
    #model.proba
    dataset = CorpusDataset([text.split()], preprocessor.get_input_features)
    model.ner_test_dataloader = DataLoader(dataset, batch_size=config.eval_batch_size)

    res = trainer.test(model)
    result = []
    for i in range(len(res['gt_labels'])):
        sentence = ''.join(dataset.sentences[i])
        pred_sentence = ''
        for j in range(len(res['pred_labels'][i])):
            # print(sentence[j]+ '==' + res['pred_labels'][i][j])
            if 'B' == res['pred_labels'][i][j] and j > 0:
                pred_sentence += ' ' + sentence[j]
            else:
                pred_sentence += sentence[j]

        print('orig:' + ' '.join(dataset.sentences[i]))
        print('pred:' + pred_sentence)

        result.append(pred_sentence)

    return ' '.join(result)


if __name__ == "__main__":
    
    spacing('내가 니시바다리가? 내가여기서 No.1인데')
    spacing('한동안서울은 강북이었습니다. 서울이 커졌죠?')
    spacing('아니요. 그렇지 않고요. 언제나 새롭습니다')
    spacing('지난주 하락에 따른 반발 매수세가 유입됐다. 현지시간으로 지난 20일 미국 뉴욕증시에서 3대 지수가 일제히 호조를 보인 점도 투자 심리를 뒷받침하고 있다. 특히 자동차 관련 종목에 매수세가 일었다. 지난주 감산 발표로 타격을 입은 토요타자동차는 이날 3.62%, 자동차 부품 업체 덴소는 7.63% 오르고 있다.')