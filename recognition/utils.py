import time
import torch
import torch.nn as nn
import torch.nn.functional as F
 
 
class DiceCELoss(nn.Module):
 
    def __init__(self, num_classes, ds_weights=(1.0, 0.5, 0.25), eps=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.ds_weights = ds_weights
        self.eps = eps
 
    def _single(self, logits, target):
        probs = torch.softmax(logits, dim=1)
        onehot = F.one_hot(target, self.num_classes).permute(0, 4, 1, 2, 3).float()
        dims = (0, 2, 3, 4)
        inter = (probs * onehot).sum(dims)
        denom = probs.sum(dims) + onehot.sum(dims)
        dice = (2 * inter + self.eps) / (denom + self.eps)
        return (1 - dice.mean()) + F.cross_entropy(logits, target)
 
    def forward(self, outputs, target):
        if isinstance(outputs, (list, tuple)):
            w = self.ds_weights[:len(outputs)]
            total = sum(wi * self._single(o, target) for wi, o in zip(w, outputs))
            return total / sum(w)
        return self._single(outputs, target)
 
 
def dice_iou_per_class(pred, target, num_classes):

    B = pred.shape[0]
    dice = torch.full((B, num_classes), float("nan"))
    iou = torch.full((B, num_classes), float("nan"))
    for c in range(num_classes):
        p = (pred == c).flatten(1)
        t = (target == c).flatten(1)
        inter = (p & t).sum(1).float()
        p_sum, t_sum = p.sum(1).float(), t.sum(1).float()
        union = p_sum + t_sum - inter
        present = (p_sum + t_sum) > 0
        dice[present, c] = (2 * inter / (p_sum + t_sum))[present].cpu()
        iou[present, c] = (inter / union)[present].cpu()
    return dice, iou
 
 
def slice_thirds_dice(pred, gt, label, axis=2):
    p = torch.movedim(pred == label, axis, 0)
    g = torch.movedim(gt == label, axis, 0)
    present = g.flatten(1).any(1).nonzero().flatten()
    if present.numel() == 0:
        return [float("nan")] * 3
    lo, hi = present.min().item(), present.max().item() + 1
    n = hi - lo
    edges = [lo + round(n * k / 3) for k in range(4)]
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        ps, gs = p[a:b], g[a:b]
        denom = ps.sum().item() + gs.sum().item()
        out.append(2 * (ps & gs).sum().item() / denom if denom > 0 else float("nan"))
    return out
 
 
def third_slice_indices(gt, label, axis=2):
    g = torch.movedim(gt == label, axis, 0)
    present = g.flatten(1).any(1).nonzero().flatten()
    if present.numel() == 0:
        mid = gt.shape[axis] // 2
        return [mid, mid, mid]
    lo, hi = present.min().item(), present.max().item() + 1
    n = hi - lo
    edges = [lo + round(n * k / 3) for k in range(4)]
    return [(a + max(a, b - 1)) // 2 for a, b in zip(edges[:-1], edges[1:])]
 
 
def nanmean(values):
    t = torch.as_tensor(values, dtype=torch.float32)
    t = t[~torch.isnan(t)]
    return t.mean().item() if t.numel() else float("nan")

 
@torch.no_grad()
def evaluate(model, loader, device, num_classes, prostate_label=5, slice_axis=2,
             criterion=None, return_per_case=False):
    model.eval()
    dices, ious, thirds, losses, times, per_case = [], [], [], [], [], []
    for img, lab in loader:
        img, lab = img.to(device), lab.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = model(img)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        if criterion is not None:
            losses.append(criterion(logits, lab).item())
        pred = logits.argmax(1)
        d, i = dice_iou_per_class(pred, lab, num_classes)
        dices.append(d)
        ious.append(i)
        for b in range(pred.shape[0]):
            th = slice_thirds_dice(pred[b].cpu(), lab[b].cpu(), prostate_label, slice_axis)
            thirds.append(th)
            per_case.append({"dice": d[b].tolist(), "thirds": th})
    dices, ious = torch.cat(dices), torch.cat(ious)
    thirds_t = torch.tensor(thirds)
    result = {
        "dice_per_class": [nanmean(dices[:, c]) for c in range(num_classes)],
        "iou_per_class": [nanmean(ious[:, c]) for c in range(num_classes)],
        "prostate_dice_thirds": [nanmean(thirds_t[:, k]) for k in range(3)],
        "mean_inference_s": sum(times) / len(times),
        "loss": sum(losses) / len(losses) if losses else None,
    }
    result["mean_fg_dice"] = nanmean(result["dice_per_class"][1:])
    if return_per_case:
        result["per_case"] = per_case
    return result
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)