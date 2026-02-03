"""Utility metrics and helper functions"""

import torch


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions
    
    Args:
        output: Model output logits [batch_size, num_classes]
        target: Ground truth labels [batch_size]
        topk: Tuple of k values
    
    Returns:
        List of top-k accuracies
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(state, filename='checkpoint.pth'):
    """Save checkpoint"""
    torch.save(state, filename)


def load_checkpoint(filename, model, optimizer=None):
    """Load checkpoint"""
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def get_learning_rate(optimizer):
    """Get current learning rate from optimizer"""
    for param_group in optimizer.param_groups:
        return param_group['lr']
    return None


if __name__ == "__main__":
    # Test metrics
    output = torch.randn(4, 10)
    target = torch.tensor([0, 1, 2, 3])
    
    top1, top5 = accuracy(output, target, topk=(1, 5))
    print(f"Top-1 Accuracy: {top1.item():.2f}%")
    print(f"Top-5 Accuracy: {top5.item():.2f}%")
    
    # Test AverageMeter
    meter = AverageMeter()
    for i in range(10):
        meter.update(i)
    print(f"Average: {meter.avg:.2f}")
