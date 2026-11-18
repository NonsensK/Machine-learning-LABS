"""CNNs and fixed-mask / physically compacted pruning."""
import copy
import torch
from torch import nn


class CNN(nn.Module):
    def __init__(self, channels=(16, 32), hidden=640):
        super().__init__()
        self.channels = tuple(channels)
        self.hidden = hidden
        self.conv1 = nn.Conv2d(1, channels[0], 3, padding=1)
        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, padding=1)
        self.fc1 = nn.Linear(channels[1] * 7 * 7, hidden)
        self.fc2 = nn.Linear(hidden, 10)

    def forward(self, x):
        x = nn.functional.max_pool2d(self.conv1(x).relu(), 2)
        x = nn.functional.max_pool2d(self.conv2(x).relu(), 2)
        return self.fc2(self.fc1(x.flatten(1)).relu())


def prune_weights(model, fraction, method):
    """Exact floor(fraction*N) pruning, with deterministic tie handling."""
    result = copy.deepcopy(model)
    weights = {name: value for name, value in result.named_parameters() if name.endswith("weight")}
    masks = {}
    if method == "global":
        flat = torch.cat([value.detach().abs().flatten() for value in weights.values()])
        keep = torch.ones_like(flat)
        keep[torch.argsort(flat, stable=True)[:int(fraction * flat.numel())]] = 0
        offset = 0
        for name, weight in weights.items():
            masks[name] = keep[offset:offset + weight.numel()].reshape_as(weight)
            offset += weight.numel()
    elif method == "layer":
        for name, weight in weights.items():
            flat = weight.detach().abs().flatten()
            keep = torch.ones_like(flat)
            keep[torch.argsort(flat, stable=True)[:int(fraction * flat.numel())]] = 0
            masks[name] = keep.reshape_as(weight)
    else:
        raise ValueError(method)
    apply_masks(result, masks)
    return result, masks


@torch.no_grad()
def apply_masks(model, masks):
    for name, parameter in model.named_parameters():
        if name in masks:
            parameter.mul_(masks[name])


def compact_filters(model, fraction):
    """Rank convolution output filters by L2 norm and rebuild smaller layers."""
    keep1 = torch.argsort(model.conv1.weight.detach().flatten(1).norm(dim=1),
                         descending=True, stable=True)[:max(1, int(model.channels[0]*(1-fraction)))].sort().values
    keep2 = torch.argsort(model.conv2.weight.detach().flatten(1).norm(dim=1),
                         descending=True, stable=True)[:max(1, int(model.channels[1]*(1-fraction)))].sort().values
    result = CNN((len(keep1),len(keep2)),model.hidden)
    columns = (keep2[:,None]*49 + torch.arange(49)).flatten()
    with torch.no_grad():
        result.conv1.weight.copy_(model.conv1.weight[keep1])
        result.conv1.bias.copy_(model.conv1.bias[keep1])
        result.conv2.weight.copy_(model.conv2.weight[keep2][:,keep1])
        result.conv2.bias.copy_(model.conv2.bias[keep2])
        result.fc1.weight.copy_(model.fc1.weight[:,columns])
        result.fc1.bias.copy_(model.fc1.bias)
        result.fc2.load_state_dict(model.fc2.state_dict())
    return result


def distillation_loss(logits, teacher_logits, labels, alpha, temperature):
    """Manual's KL(student || teacher), including T squared scaling."""
    log_student = torch.log_softmax(logits / temperature, dim=1)
    log_teacher = torch.log_softmax(teacher_logits.detach() / temperature, dim=1)
    kl = (log_student.exp() * (log_student - log_teacher)).sum(dim=1).mean()
    ce = nn.functional.cross_entropy(logits, labels)
    return alpha * ce + (1-alpha) * temperature**2 * kl


def macs(model):
    c1, c2 = model.channels
    return 28*28*c1*9 + 14*14*c2*c1*9 + c2*49*model.hidden + model.hidden*10
