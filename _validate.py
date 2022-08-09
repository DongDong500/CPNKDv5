import torch
import torch.nn as nn
import criterion

def validate(model, loader, devices, metrics, loss_type,**kwargs):
    
    costfunction = criterion.get_criterion.__dict__[loss_type](**kwargs)

    model.eval()
    metrics.reset()

    running_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):

            images = images.to(devices)
            labels = labels.to(devices)

            outputs = model(images)
            probs = nn.Softmax(dim=1)(outputs)
            preds = torch.max(probs, 1)[1].detach().cpu().numpy()
            target = labels.detach().cpu().numpy()

            loss = costfunction(outputs, labels)

            metrics.update(target, preds)
            running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    score = metrics.get_results()

    return score, epoch_loss