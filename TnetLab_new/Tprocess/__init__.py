import torch
import torch.nn.functional as F
from data_process import rgb_Norm, pic_Augment

#  AISD     : mean=[0.463, 0.475,0.451],std=[0.223, 0.211, 0.206]
#  SBU paper: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
      # resize416: [0.493, 0.482, 0.439], [0.258, 0.242, 0.257])]
      # orig size: [0.491, 0.480, 0.439], [0.264, 0.248, 0.263])
#  ISTD     : mean=[0.517, 0.514, 0.492], std=[0.186, 0.173, 0.181]
#  ISIC2017 : mean=[0.723, 0.616, 0.569], std=[0.169, 0.177, 0.197]

def bm_bce(pred, target, reduction='mean'):
    ''' balance maintain bce.
        Balance the ratios of different class.
        Maintain the whole quantity  '''

    assert (pred.size() == target.size())
    pos = torch.eq(target, 1).float()
    neg = torch.eq(target, 0).float()

    num_pos = torch.sum(pos)
    num_neg = torch.sum(neg)
    num_total = num_pos + num_neg
    assert num_total == torch.prod(torch.tensor(target.size()))

    if num_pos == 0 or num_neg == 0:
        return F.binary_cross_entropy_with_logits(pred, target, reduction=reduction)

    else:
        alpha = num_total / (2 * num_pos)
        beta = num_total / (2 * num_neg)
        weights = alpha * pos + beta * neg
        return F.binary_cross_entropy_with_logits(pred, target, weights, reduction=reduction)


