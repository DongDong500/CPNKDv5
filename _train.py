import numpy as np
import torch
import torch.nn as nn

import criterion

def get_class_weight(lbls):
    '''
    compute class weight (only binary)
        Args:
            lbls (numpy array)
        Returns:
            weight (numpy array)
    '''
    weights = lbls.sum() / (lbls.shape[0] * lbls.shape[1] * lbls.shape[2])
    
    if weights < 0 or weights > 1:
        raise Exception (f'weights: {weights} for cross entropy is wrong')

    return [weights, 1 - weights]

def train(model, loader, devices, metrics, loss_type, 
            optimizer, scheduler, **kwargs):
    
    costfunction = criterion.get_criterion.__dict__[loss_type](**kwargs)

    model.train()
    metrics.reset()
    running_loss = 0.0

    for i, (images, labels) in enumerate(loader):
        images = images.to(devices)
        labels = labels.to(devices)
        
        outputs = model(images)
        probs = nn.Softmax(dim=1)(outputs)
        preds = torch.max(probs, 1)[1].detach().cpu().numpy()

        if loss_type == 'entropydice_loss':
            class_weights = torch.tensor(get_class_weight(labels.detach().cpu().numpy()), dtype=torch.float32).to(devices)
            costfunction.update_weight(weight=class_weights)
        elif loss_type == 'dice_loss':
            pass
        elif loss_type == 'kd_loss':
            raise NotImplementedError
        elif loss_type == 'gp_loss':
            raise NotImplementedError
        else:
            raise Exception (f'{loss_type} is not option')

        optimizer.zero_grad()
        loss = costfunction(outputs, labels)
        loss.backward()

        optimizer.step()
        
        metrics.update(labels.detach().cpu().numpy(), preds)
        running_loss += loss.item() * images.size(0)
    
    scheduler.step()
    epoch_loss = running_loss / len(loader.dataset)
    score = metrics.get_results()
    
    return score, epoch_loss



if __name__ == "__main__":

    print(criterion.get_criterion.__dict__['gp_loss']())
