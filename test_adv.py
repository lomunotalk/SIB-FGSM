from pretrained.resnet import resnet20
from pretrained.resnet import resnet44
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from d2l import torch as d2l
import os as os


# 导入预训练模型，'state_dict' 在all_save内部
all_save = torch.load(
    "F:pretrained\pretrained_models\\resnet20-12fca82f.th",
    map_location='cpu')

# 创建空模型
res20 = resnet20()
res20 = nn.DataParallel(res20)
# 把参数导入到模型中
res20.load_state_dict(all_save['state_dict'], False)

# 加载数据
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([transforms.ToTensor(), normalize, ])



# eps_list = [1, 2, 3, 4, 5, 6, 7, 8, 16, 32]
# alpha_list = [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7],
#               [1, 2, 3, 4, 5, 6, 7, 8],
#               [1, 2, 3, 8, 12, 16], [2, 4, 8, 12, 16, 24, 32]]

# 测试参数空间 集合。
eps_list = [12]
alpha_list = [[1]]


path_root = "F:all\\dsib"

for i in range(0, len(eps_list)):
    eps_numerator = eps_list[i]
    alpha_numerator_list = alpha_list[i]
    for j in range(0, len(alpha_numerator_list)):
        eps = eps_numerator
        alpha = alpha_numerator_list[j]
        path = path_root + "\\" + str(eps) + "-" + str(alpha)
        test_data = datasets.ImageFolder(path,transform=transform)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=False, num_workers=0)    # num_workers=args.workers, pin_memory=True)
        # 测试成功率
        test_acc = d2l.evaluate_accuracy_gpu(res20, test_loader)
        print(str(eps) + "-" + str(alpha)+"::"+str(format(test_acc, '.3f')))  # ori = 1

        # print(test_acc)  # adv = 0.2789473684210526  所有的都生成了。