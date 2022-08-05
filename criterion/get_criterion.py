try: 
    from entropydice import EntropyDiceLoss
    from focal import FocalLoss
    from dice import DiceLoss
    from crossentropy import CrossEntropyLoss
    from kdloss import KDLoss
    from gploss import GPLoss
except:
    from .entropydice import EntropyDiceLoss
    from .focal import FocalLoss
    from .dice import DiceLoss
    from .crossentropy import CrossEntropyLoss
    from .kdloss import KDLoss
    from .gploss import GPLoss

def entropydice_loss(**kwargs):

    return EntropyDiceLoss()

def dice_loss(**kwargs):

    return DiceLoss()

def kd_loss(opts, **kwargs):
    
    return KDLoss(alpha=opts.alpha, temperature=opts.T)

def gp_loss(**kwargs):
    
    return GPLoss()


if __name__ == "__main__":

    import torch
    import torch.nn as nn
    
    loss = dice_loss()

    m = nn.Softmax()

    s_input = torch.randn((5, 2, 256, 256), requires_grad=True)
    # t_input = torch.rand((5, 2, 256, 256), requires_grad=True)
    target = torch.randint(0, 2, (5, 256, 256))

    print("s_input: \t{}".format(s_input.size()))
    # print("t_input: \t{}".format(t_input.size()))
    print("target: \t{}".format(target.size()))
    print(loss(s_input, target))