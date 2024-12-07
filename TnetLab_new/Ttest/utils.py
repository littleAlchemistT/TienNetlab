import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from torchvision import transforms
from PIL import Image


class testDataset(Dataset):
    def __init__(self, data_root , subfile, resize=None,img_norm=None
                 , data_ext=('.tif', '.png', '.jpg')):
        self.data_root=data_root
        self.subfile=subfile
        self.data_ext=data_ext

        self.resize=resize
        self.img_norm = img_norm  # with or without norm
        self.imgs = self._dataPathList()


    def _dataPathList(self):
        img_name_lst = [f for f in os.listdir(os.path.join(self.data_root, self.subfile[0])) if f.endswith(self.data_ext)]
        label_name_lst=(f for f in os.listdir(os.path.join(self.data_root, self.subfile[1])) if f.endswith(self.data_ext))

        pair_data_path = []
        for name in label_name_lst:
            if name in img_name_lst:
                pair_data_path.append((os.path.join(self.data_root, self.subfile[0], name)
                              , os.path.join(self.data_root, self.subfile[1], name)))
        
        assert len(pair_data_path)>0, 'empty data'

        return pair_data_path


    def __getitem__(self, index):
        img_path, gt_path = self.imgs[index]
        img_name = os.path.basename(img_path)
        img = Image.open(img_path).convert('RGB')
        target = Image.open(gt_path).convert('L')
        if self.resize:
            img = transforms.Resize(self.resize)(img)
        if self.img_norm:
            img = self.img_norm(transforms.ToTensor()(img))
        else:
            img = transforms.ToTensor()(img)
        
        target = transforms.ToTensor()(target)
        sample = {'image': img, 'label': target, 'img_name':img_name}
        return sample

    def __len__(self):
        return len(self.imgs)


def testDataLoader(data_root , subfile, resize=None,img_norm=None):
    dataset=testDataset(data_root , subfile, resize=resize,img_norm=img_norm)

    return DataLoader(dataset, batch_size=1)



###########################################################################
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

    
    
################################ loss #####################################
def bm_bce(pred, target,reduction='mean'):
    ''' balance maintain bce.
        Balance the ratios of different class.
        Maintain the whole quantity  '''
    
    assert(pred.shape == target.shape)
    tar_res=torch.prod(torch.tensor(target.shape))
    pos = (target>=0.5).float()
    neg = (target<0.5).float()
    num_pos = torch.sum(pos)
    num_neg = torch.sum(neg)
    num_total = num_pos + num_neg


    if num_total!=tar_res:
        assert False, f'Input res != mtrx total. mtrx total={num_total}, input res={tar_res}'
        
    if num_total==0:
        assert False, 'Input reslutions = 0 ! '

    elif num_pos==0 or num_neg==0:
        return F.binary_cross_entropy_with_logits(pred, target, reduction=reduction)
    
    else:
        alpha = num_total/(2*num_pos)
        beta = num_total/(2*num_neg)
        
        weights = alpha * pos + beta * neg
        
        return F.binary_cross_entropy_with_logits(pred, target, weights, reduction=reduction)


####################  crf  ####################

# def _sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# def crf_refine(img, annos):
#     assert img.dtype == np.uint8
#     assert annos.dtype == np.uint8
#     assert img.shape[:2] == annos.shape , f"img shape={img.shape}, pred shape={annos.shape} "

#     # img and annos should be np array with data type uint8

#     EPSILON = 1e-8

#     M = 2  # salient or not
#     tau = 1.05
#     # Setup the CRF model
#     d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], M)

#     anno_norm = annos / 255.

#     n_energy = -np.log((1.0 - anno_norm + EPSILON)) / (tau * _sigmoid(1 - anno_norm))
#     p_energy = -np.log(anno_norm + EPSILON) / (tau * _sigmoid(anno_norm))

#     U = np.zeros((M, img.shape[0] * img.shape[1]), dtype='float32')
#     U[0, :] = n_energy.flatten()
#     U[1, :] = p_energy.flatten()

#     d.setUnaryEnergy(U)

#     d.addPairwiseGaussian(sxy=3, compat=3)
#     d.addPairwiseBilateral(sxy=60, srgb=5, rgbim=img, compat=5)

#     # Do the inference
#     infer = np.array(d.inference(1)).astype('float32')
#     res = infer[1, :]

#     res = res * 255
#     res = res.reshape(img.shape[:2])
#     return res.astype('uint8')
