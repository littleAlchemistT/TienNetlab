import random
from PIL import Image

#-------------------- data: [ PIL.Image ] --------------------

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img_pairs):
        input_size=img_pairs[0].size
        for i in img_pairs:
            assert i.size == input_size, "Data pairs have different size"

        for t in self.transforms:
            img_pairs = t(img_pairs)

        return  img_pairs



class Resize(object):
    def __init__(self, size):
        # self.size = tuple(reversed(size))  # size: (h, w)
        self.size = size  # (w,h)

    def __call__(self, img_list):
        input_size=img_list[0].size
        out_img=[]
        for im in img_list:
            assert im.size == input_size, "Data pairs have different size"
            out_img.append(im.resize(self.size, Image.BILINEAR))
        return out_img # a list




class RandomHorizontallyFlip(object):
    def __call__(self, img_list):
        if random.random() < 0.5:
            out_img=[]
            for i in img_list:
                out_img.append(i.transpose(Image.FLIP_LEFT_RIGHT))
            return out_img  # a list
        return img_list # a list




class RandomCrop(object):
    '''After resize.'''
    def __init__(self, crop_ratio=0.9):
        self.crop_ratio=crop_ratio
    def __call__(self, img_list):
        if random.random()>0.5:
            w, h= img_list[0].size
            height = h*self.crop_ratio
            width = w*self.crop_ratio
            w_start = random.randint(0, w - width)
            h_start = random.randint(0, h - height)
            out_list=[]
            for img in img_list:
                out_list.append(img.crop(h_start,w_start,h_start+height,w_start+width))
            return out_list

        return img_list




