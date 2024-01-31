# 把第二高的类别，分成不同比例区间的。


import torch
# import resnet
from d2l import torch as d2l
from torch.autograd import Variable
import torch.nn as nn
import argparse
import os
import shutil
import time
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from pretrained.resnet import resnet20
from pretrained.resnet import resnet44
from pretrained.resnet import resnet110
from pretrained.resnet import resnet1202
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import gen_adv
from transform_convert import transform_convert

classify_detail = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
def save(classify_list, img_tensors, transform, targets,batch_idx):
    path_root = "F:\\Cifar10-all\\clean\\Cifar-devided-byper-from-500-resnet44\\per-"
    # 已经创建好的文件夹。per-1   到  per-9  。并且每一个per下都有0-9的空文件夹，所需的文件夹都创建好了

    for i in range(0, len(classify_list)):
        same_per_list = classify_list[i]
        if len(same_per_list) == 0:
            continue
        path_per = path_root + str(i)
        for j in range(0, len(same_per_list)):
            img_index = same_per_list[j]  # img_index 0-15
            img = transform_convert(img_tensors[img_index], transform)  # 总共进行16次 一个batch一共16个照片
            # 图片保存地址
            target = targets[img_index].item()
            target_save_path = path_per + "\\" + str(target)+"\\"+ str(batch_idx)+"-"+str(i)+"-"+str(j)+".png"
            img.save(target_save_path)
            classify_detail[i][target] = classify_detail[i][target] + 1 # todo #


# 导入预训练模型，'state_dict' 在all_save内部
all_save = torch.load(
    "F:pretrained\pretrained_models\\resnet44-014dd654.th",
    map_location='cpu')

# 创建空模型
# model = resnet20()
model = resnet44()
model = nn.DataParallel(model)
# 把参数导入到模型中
model.load_state_dict(all_save['state_dict'], False)

# 加载数据
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([transforms.ToTensor(), normalize])
test_data = datasets.ImageFolder("F:Cifar10-all\\clean\\Cifar10-500",
                                 transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=False,
                                          num_workers=0)  # num_workers=args.workers, pin_memory=True)

device = 'cpu'
model.eval()  # ln dropout 不起作用
criterion = nn.CrossEntropyLoss()
# 创建存放按照per分类照片路径,并且在路径中，创建了10个分类文件夹（0----9）
path_root = "F:Cifar10-all\\clean\\Cifar-devided-byper-from-500-resnet44\\per-"
for i in range(0, 10):
    per_i = path_root + str(i)
    if not os.path.exists(per_i):
        os.makedirs(per_i)
    for j in range(10):
        per_i_j = per_i + "\\" + str(j)
        if not os.path.exists(per_i_j):
            os.makedirs(per_i_j)
classify_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
for idx, (images, targets) in enumerate(test_loader):
    images = Variable(images.to(device), requires_grad=False)
    targets = Variable(targets.to(device))

    y = model(images)
    sorted_class, sorted_index = torch.sort(y)
    y_second = sorted_class[:, 8]  # 第二大的
    y_first = sorted_class[:, 9]  # 第一大的
    # 直接比较不用e来算了，计算简单。
    rate = y_second / y_first
    classify_list = [[], [], [], [], [], [], [], [], [], []]
    for i in range(0, rate.shape[0]):
        rate_int_index = int(10 * rate[i].item())
        if rate_int_index<0:
            continue
        classify_list[rate_int_index].append(i)
        classify_count[rate_int_index] = classify_count[rate_int_index]+1

    save(classify_list, images, transform, targets,idx)

print(classify_count)
print(classify_detail)