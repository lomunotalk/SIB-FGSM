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
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import gen_adv

# 导入预训练模型，'state_dict' 在all_save内部
all_save = torch.load(
    "F:\pretrained_models\\resnet20-12fca82f.th",
    map_location='cpu')

# 创建空模型
res20 = resnet20()
res20 = nn.DataParallel(res20)
# 把参数导入到模型中
res20.load_state_dict(all_save['state_dict'], False)

# 加载数据
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([transforms.ToTensor(), normalize, ])
test_data = datasets.ImageFolder("F:\\Cifar10-all\\clean\\Cifar10-all", transform=transform)  # 50的干净的小数据集
# test_data = datasets.ImageFolder("F:\\AI\\ai数据集\\Cifar10-all\\clean\\Cifar-devided-byper-from-500-resnet44\\per-4", transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=False,
                                          num_workers=0)  # num_workers=args.workers, pin_memory=True)
# todo 抽取。是否用dsib或者i-fgsm，是否融合，融合方式1,2，融合比例a，b。总扰动，单次扰动，以及扰动比例和不能超过总扰动。
# 参数空间遍历
# eps_list = [1, 2, 3, 4, 5, 6, 7, 8, 16, 32]
# alpha_list = [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7],
#               [1, 2, 3, 4, 5, 6, 7, 8],
#               [1, 2, 3, 8, 12, 16], [2, 4, 8, 12, 16, 24, 32]]
# 测试参数空间 集合。
eps_list = [12]
alpha_list = [[1]]
sxb = 8  # 1==9   8==第二大的概率。9==第一大的概率。
targeted_sxb = True  # sib-fgsm 调参  # 用基本迭代版本生产1000个adv样本      targeted_sxb=是否开启增强第二个。true是开启
for i in range(0, len(eps_list)):
    eps_numerator = eps_list[i]
    alpha_numerator_list = alpha_list[i]
    for j in range(0, len(alpha_numerator_list)):
        gen_adv.basic_iteration(test_loader, res20, transform, eps=eps_numerator, alpha=alpha_numerator_list[j],
                                targeted_sxb=True, sxb=sxb)

# targeted_sxb=False #i-fgsm  调参


for i in range(0, len(eps_list)):
    eps_numerator = eps_list[i]
    alpha_numerator_list = alpha_list[i]
    for j in range(0, len(alpha_numerator_list)):
        gen_adv.basic_iteration(test_loader, res20, transform, eps=eps_numerator, alpha=alpha_numerator_list[j],
                                targeted_sxb=False)
print()
