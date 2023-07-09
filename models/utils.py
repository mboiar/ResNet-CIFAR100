from typing import Optional, Dict

import torch.nn as nn
import torch
import torch.nn.functional as F


def get_probs(
    net: nn.Module, data: torch.Tensor, k: int = 5,
    add_softmax: Optional[bool] = False
) -> Dict[int, float]:
    """Run inference on data and return top-k probabilities."""

    if len(data.shape) == 3:
        data = data[None, :, :, :]

    output = net(data).data if add_softmax == False else F.softmax(net(data).data, dim=1)

    val, ind = torch.topk(output, k=k)
    val, ind = val.numpy(), ind.numpy()
    return {ind[0][i].item():val[0][i].item() for i in range(val.size)}
