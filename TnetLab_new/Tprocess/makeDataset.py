import os
import random
import torch.utils.data as data
from PIL import Image
import torch
from utils.util import cal_subitizing


NO_LABEL = -1

class Dataset4Loader(data.Dataset):
    def __init__(self, data_root , subfile, img_tensor, target_tensor, data_aug=None
                 , data_ext=('.tif', '.png', '.jpg'), if_edge=False, if_prop=False
                 ,if_semi=False, unlb_root=None, lb_rto=1., seed=1337
                 , subitizing_threshold=8, subitizing_min_size_per=0.005
                 ):
        self.data_root=data_root
        self.subfile=subfile
        self.data_ext=data_ext
        self.if_edge = if_edge
        self.if_prop = if_prop
        self.if_semi=if_semi
        self.unlb_root=unlb_root
        self.lb_rto=lb_rto
        self.seed=seed
        self.subitizing_threshold = subitizing_threshold
        self.subitizing_min_size_per = subitizing_min_size_per

        self.data_aug = data_aug
        self.img_tensor = img_tensor  # with or without norm
        self.target_tensor = target_tensor
        self.imgs = self._dataPathList()

    def _dataPathList(self):
        # (root, subFile, data_ext, if_edge=True, if_semi=False, unlabeled_root=None)
        img_name_list = (f for f in os.listdir(os.path.join(self.data_root, self.subfile[0])) if f.endswith(self.data_ext))
        label_name_list = [f for f in os.listdir(os.path.join(self.data_root, self.subfile[1])) if f.endswith(self.data_ext)]
        assert len(label_name_list) > 0, "data list is empty"

        data_list, data_unlabeled_list = [],[]
        n_label, n_img = 0, 0
        for idx, img_name in enumerate(img_name_list):
            n_img += 1
            if img_name in label_name_list:
                n_label += 1
                if self.if_edge:
                    data_list.append((os.path.join(self.data_root, self.subfile[0], img_name),
                                      os.path.join(self.data_root, self.subfile[1], img_name),
                                      os.path.join(self.data_root, self.subfile[2], img_name)))
                else:
                    data_list.append((os.path.join(self.data_root, self.subfile[0], img_name),
                                      os.path.join(self.data_root, self.subfile[1], img_name)))
            else:  # no label
                if self.if_edge:
                    data_unlabeled_list.append(
                        (os.path.join(self.data_root, self.subfile[0], img_name), -1, -1))
                else:
                    data_unlabeled_list.append(
                        (os.path.join(self.data_root, self.subfile[0], img_name), -1))
        if self.if_semi:
            if self.unlb_root is not None:
                print(f'unla root: {self.unlb_root}')
                if self.if_edge:
                    data_unlabeled_list = [(os.path.join(self.unlb_root, self.subfile[0], f), -1, -1) for f in
                                           os.listdir(os.path.join(self.unlb_root, self.subfile[0])) if
                                           f.endswith(self.data_ext)]
                else:
                    data_unlabeled_list = [(os.path.join(self.unlb_root, self.subfile[0], f), -1) for f in
                                           os.listdir(os.path.join(self.unlb_root, self.subfile[0])) if
                                           f.endswith(self.data_ext)]
            else:
                print(f'remove labels...labeled ratio = {self.lb_rto}')
                if n_img * self.lb_rto < n_label:
                    print('remove label for semi ...')
                    random.seed(self.seed)
                    unlb_idx = random.sample(range(len(data_list)), n_label - int(n_img * self.lb_rto))
                    for idx in unlb_idx:  # random remove targets
                        data_list[idx] = list(data_list[idx])
                        if self.if_edge:
                            data_list[idx][1:] = -1, -1
                        else:
                            data_list[idx][1] = -1
                        data_list[idx] = tuple(data_list[idx])

            data_list=data_list+data_unlabeled_list
            random.shuffle(data_list)
        del label_name_list

        return data_list



    def __getitem__(self, index):
        if self.if_edge:
            img_path, gt_path, edge_path = self.imgs[index]
        else:
            img_path, gt_path = self.imgs[index]
        img_name=os.path.basename(img_path)
        img = Image.open(img_path).convert('RGB')
        if gt_path == -1:  # unlabeled data
            img = self.img_tensor(self.data_aug([img])[0])
            # ↓ fake label to make sure pytorch inner check
            target = torch.zeros((img.shape[-2], img.shape[-1])).unsqueeze(0)
            if self.if_prop and self.if_edge:
                edge = torch.zeros((img.shape[-2], img.shape[-1])).unsqueeze(0)
                prop = torch.ones(1)
                sample = {'image': img, 'label': target, 'proportion': prop, 'edge': edge, 'img_name':img_name}
            elif self.if_prop:
                prop = torch.ones(1)
                sample = {'image': img, 'label': target, 'proportion': prop, 'img_name':img_name}
            elif self.if_edge:
                edge = torch.zeros((img.shape[-2], img.shape[-1])).unsqueeze(0)
                sample = {'image': img, 'label': target, 'edge': edge, 'img_name':img_name}
            else:
                sample = {'image': img, 'label': target, 'img_name':img_name}

        else:  # labeled data
            target = Image.open(gt_path).convert('L')
            if self.if_edge:
                edge = Image.open(edge_path).convert('L')
                if self.data_aug:
                    img, target, edge = self.data_aug([img, target, edge])
                edge = self.target_tensor(edge)
            else:
                if self.data_aug:
                    img, target = self.data_aug([img, target])

            if self.if_prop:
                prop, percentage = cal_subitizing(target, threshold=self.subitizing_threshold,
                                                            min_size_per=self.subitizing_min_size_per)
                prop = torch.Tensor([prop])

                # number_per = cal_prop(target)
            img = self.img_tensor(img)
            target = self.target_tensor(target)

            if self.if_prop and self.if_edge:
                sample = {'image': img, 'label': target, 'proportion': prop, 'edge': edge, 'img_name':img_name}
            elif self.if_prop:
                sample = {'image': img, 'label': target, 'proportion': prop, 'img_name':img_name}
            elif self.if_edge:
                sample = {'image': img, 'label': target, 'edge': edge, 'img_name':img_name}
            else:
                sample = {'image': img, 'label': target, 'img_name':img_name}

        return sample

    def __len__(self):
        return len(self.imgs)



def relabel_dataset(dataset, edge_able=False):
    unlabeled_idxs = []
    for idx in range(len(dataset.imgs)):
        if not edge_able:
            path, label = dataset.imgs[idx]
        else:
            path, label, edge = dataset.imgs[idx]
        if label == -1:
            unlabeled_idxs.append(idx)
    labeled_idxs = sorted(set(range(len(dataset.imgs))) - set(unlabeled_idxs))

    return labeled_idxs, unlabeled_idxs







