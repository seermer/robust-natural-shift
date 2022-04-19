import torch
from torch.nn import functional as F
import pandas as pd
from typing import List


@torch.no_grad()
def labeling(data, models: List[torch.nn.Module], mode: str, batch_size: int):
    for model in models:
        model.val()
    predictions = [[] for _ in range(len(models))]
    data = torch.split(data, batch_size)
    for batch in data:
        for j, model in enumerate(models):
            predictions[j].append(model(batch))
    predictions = [torch.vstack(val) for val in predictions]
    num_classes = predictions[0].size()[1]

    for i in range(len(predictions)):
        pred = torch.argmax(predictions[i], dim=-1, keepdim=True)
        pred = pd.DataFrame({'pred': pred.numpy()})
        pred['index'] = pred.index
        pred = pred[pred['pred'] != (num_classes - 1)]
        predictions[i] = pred
    predictions = pd.concat(predictions, ignore_index=True)
    predictions.drop_duplicates(inplace=True, keep=False)
    data = torch.index_select(data, dim=0, index=torch.tensor(predictions['index'].values).view(-1))
    pred_one_hot = F.one_hot(torch.tensor(predictions['pred'].values).view(-1))
    weight = torch.ones((len(data)))
    if 'concat' in mode:
        return data, pred_one_hot, weight
    elif 'weight' in mode:
        pass
