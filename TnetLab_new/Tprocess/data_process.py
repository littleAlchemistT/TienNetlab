import os
import torchvision.transforms as transforms
import torch
from PIL import Image
import cv2
import numpy as np


def rgb_Norm(root, resize=None, same_size=False, dot_num=3):

    if same_size or resize:
        if resize:
            print('Resize ......')
            img_list = [transforms.Compose([transforms.Resize((resize,resize)),transforms.ToTensor()])(Image.open(os.path.join(root, img_name)).convert('RGB')) \
                        for img_name in os.listdir(root)]
        else:
            print('Same size ......')
            img_list=[transforms.ToTensor()(Image.open(os.path.join(root,img_name)).convert('RGB'))\
                      for img_name in os.listdir(root)]

        allImg_tensor = torch.stack(img_list,dim=0)
        r_m = torch.mean(allImg_tensor[:,0,:,:])
        g_m = torch.mean(allImg_tensor[:,1,:,:])
        b_m = torch.mean(allImg_tensor[:,2,:,:])

        r_std = torch.std(allImg_tensor[:,0,:,:])
        g_std = torch.std(allImg_tensor[:,1,:,:])
        b_std = torch.std(allImg_tensor[:,2,:,:])

    else:
        print('Different size ......')
        r_m, g_m, b_m = 0, 0, 0
        r_m2, g_m2, b_m2 = 0, 0, 0
        res=0
        for img_name in os.listdir(root):
            img_path=os.path.join(root,img_name)
            img=transforms.ToTensor()(Image.open(img_path).convert('RGB'))
            img2=torch.pow(img,2)

            res+=img.shape[1]*img.shape[2]
            r_m+=torch.sum(img[0,:,:])
            g_m+=torch.sum(img[1,:,:])
            b_m+=torch.sum(img[2,:,:])

            r_m2 += torch.sum(img2[0, :, :])
            g_m2 += torch.sum(img2[1, :, :])
            b_m2 += torch.sum(img2[2, :, :])

        r_m=r_m/res
        g_m=g_m/res
        b_m=b_m/res

        r_m2 = r_m2 / res
        g_m2 = g_m2 / res
        b_m2 = b_m2 / res

        r_std = (r_m2-(r_m)**2)**0.5
        g_std = (g_m2-(g_m)**2)**0.5
        b_std = (b_m2-(b_m)**2)**0.5

    return ([round(r_m.item(),dot_num), round(g_m.item(),dot_num), round(b_m.item(),dot_num)]
            ,[round(r_std.item(),dot_num), round(g_std.item(),dot_num), round(b_std.item(),dot_num)])





def edge_detect(root_file, save_file):
    os.makedirs(save_file,exist_ok=True)

    name_lst=(f for f in os.listdir(root_file))
    for name in name_lst:
        pic=cv2.imread(os.path.join(root_file,name))
        edge=cv2.Canny(pic,80, 200)
        cv2.imwrite(os.path.join(save_file,name),edge)





def pic_Augment(source_root, save_root, scale, resize=False ,subfile=None, pave=None):
    ''' scale=256, pave=64 '''
    if not pave:
        pave = scale
    acc_posi=int(0.8*scale)

    read_path, aug_path = [], []
    if subfile:
        for fn in subfile:
            read_path.append(os.path.join(source_root, fn))
            aug_path.append(os.path.join(save_root, fn))

    else:
        read_path.append(source_root)
        aug_path.append(save_root)

    for pt in aug_path:
        os.makedirs(pt, exist_ok=True)

    imgname_list = [img_name for img_name in os.listdir(read_path[0])]
    for n, name in enumerate(imgname_list):
        counts = 1

        img_pair = []
        for f in read_path:
            img_pair.append(cv2.imread(os.path.join(f, name)))

        orig_sz = img_pair[0].shape
        print('orig pic sz : ', orig_sz)

        if orig_sz[:-1] < (scale,scale):
            if resize:
                print('Resize ', name, ': ')
                for n, pic in enumerate(img_pair):
                    aug_img = np.array(transforms.Resize((scale, scale))(transforms.ToPILImage()(pic)))
                    cv2.imwrite(os.path.join(aug_path[n]
                                             ,os.path.splitext(name)[0] + '_' + str(counts) + os.path.splitext(name)[1])
                                ,aug_img)
                counts += 1
            else:
                print('Drop the small pic ', name)

        else:
            print('Crop ', name , ': ')
            for i in range(0, orig_sz[0] - acc_posi, pave):
                for j in range(0, orig_sz[1] - acc_posi, pave):
                    for n, pic in enumerate(img_pair):
                        if i+scale>pic.shape[0] and j+scale>pic.shape[1]:
                            aug_img = pic[(-1)*scale:, (-1)*scale:, :]
                            cv2.imwrite(os.path.join(aug_path[n]
                                                     , os.path.splitext(name)[0] + '_' + str(counts) +
                                                     os.path.splitext(name)[1])
                                        , aug_img)
                        elif i+scale>pic.shape[0]:
                            aug_img = pic[(-1) * scale:, j:j + scale, :]
                            cv2.imwrite(os.path.join(aug_path[n]
                                                     , os.path.splitext(name)[0] + '_' + str(counts) +
                                                     os.path.splitext(name)[1]), aug_img)
                        elif j+scale>pic.shape[1]:
                            aug_img = pic[i:i + scale, (-1)*scale:, :]
                            cv2.imwrite(os.path.join(aug_path[n]
                                                     , os.path.splitext(name)[0] + '_' + str(counts) +
                                                     os.path.splitext(name)[1]), aug_img)
                        else:
                            aug_img = pic[i:i + scale, j:j + scale, :]
                            cv2.imwrite(os.path.join(aug_path[n]
                                                     ,os.path.splitext(name)[0] + '_' + str(counts) + os.path.splitext(name)[1]),aug_img)
                    counts += 1




# if __name__ == "__main__":
#     source_root=r'C:\Users\thea_\OneDrive\YunProject\Datasets\RSdatasets\SSAD\SSAD_Corrected'
#     save_root=r'C:\Users\thea_\OneDrive\YunProject\Datasets\RSdatasets\SSAD\SSAD_Corrected\scale256'
#     os.makedirs(save_root,exist_ok=True)
#
#     pic_Augment(source_root, save_root, scale=256, resize=False, subfile=['shadow','mask','edge'], pave=64)

