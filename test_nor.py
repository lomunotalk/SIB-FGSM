from pretrained.resnet import resnet20
from pretrained.resnet import resnet44
from pretrained.resnet import resnet56
from pretrained.resnet import resnet1202
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from d2l import torch as d2l
import os as os


# 导入预训练模型，'state_dict' 在all_save内部
all_save = torch.load(
    "F:\pretrained_models\\resnet44-014dd654.th",
    map_location='cpu')

# 创建空模型
res20 = resnet44()
res20 = nn.DataParallel(res20)
# 把参数导入到模型中
res20.load_state_dict(all_save['state_dict'], False)

# 加载数据
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([transforms.ToTensor(), normalize ])


path_root = "F:\Cifar10-all\\clean\\Cifar10-50"


test_data = datasets.ImageFolder(path_root,transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=False, num_workers=0)    # num_workers=args.workers, pin_memory=True)
# 测试成功率
test_acc = d2l.evaluate_accuracy_gpu(res20, test_loader)
print(str(format(test_acc, '.3f')))  # ori = 1

        # print(test_acc)  # adv = 0.2789473684210526  所有的都生成了。