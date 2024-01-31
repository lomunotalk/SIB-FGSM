import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import os as os


def transform_convert(img_tensor, transform):
    """
    param img_tensor: tensor
    param transforms: torchvision.transforms
    """
    if 'Normalize' in str(transform):
        normal_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform.transforms))
        mean = torch.tensor(normal_transform[0].mean, dtype=img_tensor.dtype, device=img_tensor.device)
        std = torch.tensor(normal_transform[0].std, dtype=img_tensor.dtype, device=img_tensor.device)
        img_tensor.mul_(std[:, None, None]).add_(mean[:, None, None])

    img_tensor = img_tensor.transpose(0, 2).transpose(0, 1)  # C x H x W  ---> H x W x C

    if 'ToTensor' in str(transform) or img_tensor.max() < 1:
        img_tensor = img_tensor.detach().numpy() * 255

    if isinstance(img_tensor, torch.Tensor):
        img_tensor = img_tensor.numpy()

    if img_tensor.shape[2] == 3:
        img = Image.fromarray(img_tensor.astype('uint8')).convert('RGB')
    elif img_tensor.shape[2] == 1:
        img = Image.fromarray(img_tensor.astype('uint8')).squeeze()
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_tensor.shape[2]))

    return img


from torchvision.utils import save_image
from pathlib import Path


#  根据target创建文件存放扰动后的图片 eps, alpha 用的是分子
def transform_convert_save(idx, img_tensors, transform, targets, eps, alpha, targeted_sxb=False):

    path_root = "F:\\from-normal\\all\\ifgsm"  # todo  targeted_sxb=false
    if targeted_sxb:
        path_root = "F:\\adv\\from-normal\\all\\dsib"  # todo targeted_sxb=true
    path = path_root + "\\"+str(eps)+"-"+str(alpha)
    path_exist = os.path.exists(path)
    if not path_exist:
        os.makedirs(path)
    path = path+"\\"
    with torch.no_grad():
        if targets[0] == targets[targets.shape[0] - 1]:  # todo 提前创建好文件夹，不用判断了，直接存储
            path = path + str(targets[0].item())
            path_exist = os.path.exists(path)
            if not path_exist:
                os.makedirs(path)
            for i in range(img_tensors.shape[0]):
                img = transform_convert(img_tensors[i], transform)
                a = path + "\\" + str(idx) + "-" + str(i) + ".png"
                img.save(a)
        else:
            #  里面有两类的，分别创建 路径 文件夹。
            for i in range(img_tensors.shape[0]):
                path1 = path + str(targets[i].item())
                path_exist = os.path.exists(path1)
                if not path_exist:
                    os.makedirs(path1)
                img = transform_convert(img_tensors[i], transform)
                a = path1 + "\\" + str(idx) + "-" + str(i) + ".png"
                img.save(a)



# def test_trans_imagenet()