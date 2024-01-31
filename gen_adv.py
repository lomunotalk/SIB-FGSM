import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image
import transform_convert
from utils.utils import where
import numpy as np
import torchvision.transforms as transforms


# eps=, alpha=  传入的是分子
def basic_iteration(test_loader, model, transform, eps, alpha, iteration=1, x_val_min=-1, x_val_max=1,
                    targeted_sxb=False,sxb=8):
    device = 'cpu'
    # eps = 0.3  # 增大扰动幅度
    model.eval()
    criterion = nn.CrossEntropyLoss()
    for idx, (images, targets) in enumerate(test_loader):
        images = Variable(images.to(device),
                          requires_grad=True)  # 老写法
        targets = Variable(targets.to(device))
        x_adv = dsib2_i_fgsm(model, images, targets, targeted_sxb=targeted_sxb, eps=eps, alpha=alpha,
                                 iteration=iteration,
                                 x_val_min=x_val_min, x_val_max=x_val_max, criterion=criterion,sxb=sxb)
        # 保存
        transform_convert.transform_convert_save(idx, x_adv, transform, targets, eps=eps, alpha=alpha,
                                                 targeted_sxb=targeted_sxb)


import math

def dsib2_i_fgsm(net, x, y, targeted_sxb=False, eps=1, alpha=1, iteration=1, x_val_min=-1, x_val_max=1,
           criterion=None,sxb=8):
    x_adv = Variable(x.data, requires_grad=True)  # 老版本的tensor 不可以求导，需要包装秤variable。新版本variable和tensor合并到一起了。
    # iteration = math.ceil((1.25*eps) / alpha)  # 把 eps 平均分配到每个iteration
    iteration = math.ceil(eps / alpha)  # 把 eps 平均分配到每个iteration
    # eps = format(eps / 255, '.2f')  # 转换成 小数形式
    eps = float(format(eps / 255, '.4f'))  # 转换成 小数形式
    alpha = alpha / 255 # 转换成 小数形式

    for i in range(iteration):
        h_adv = net(x_adv)
        if targeted_sxb:  # 目标为第二高的分类 if i<=1 动态融合
            _, sorted_index = torch.sort(h_adv)
            # 向量加速todo
            y_second = sorted_index[:, sxb]
            for j in range(0, y.shape[0]):
                if y[j] != sorted_index[j][9]:  # 只要正确类不在最第一大概率的位置上了，就取第一大概率。第一大概率降到第二后，又被提升起来了。
                    y_second[j] = sorted_index[j][9]
            y_second = Variable(y_second.data, requires_grad=False)
            cost = criterion(h_adv, y_second)
        else:
            cost = -criterion(h_adv, y)

        net.zero_grad()
        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)

        cost.backward()
        x_adv.grad.sign_()
        grad = x_adv.grad
        x_adv = x_adv - alpha * grad
        x_adv = where(x_adv > x + eps, x + eps, x_adv)
        x_adv = where(x_adv < x - eps, x - eps, x_adv)
        x_adv = torch.clamp(x_adv, x_val_min, x_val_max)
        x_adv = Variable(x_adv.data, requires_grad=True)
    return x_adv



























def dsib_fgsm(net, x, y, targeted_sxb=False, eps=1, alpha=1, iteration=1, x_val_min=-1, x_val_max=1,
           criterion=None,sxb=8):
    x_adv = Variable(x.data, requires_grad=True)  # 老版本的tensor 不可以求导，需要包装秤variable。新版本variable和tensor合并到一起了。
    # iteration = math.ceil((1.25*eps) / alpha)  # 把 eps 平均分配到每个iteration
    iteration = math.ceil(eps / alpha)  # 把 eps 平均分配到每个iteration
    # eps = format(eps / 255, '.2f')  # 转换成 小数形式
    eps = float(format(eps / 255, '.4f'))  # 转换成 小数形式
    alpha = alpha / 255 # 转换成 小数形式

    for i in range(iteration):
        h_adv = net(x_adv)
        if targeted_sxb:  # 目标为第二高的分类
            _, sorted_index = torch.sort(h_adv)
            # 向量加速todo
            y_second = sorted_index[:, sxb]
            for j in range(0, y.shape[0]):
                if y[j] != sorted_index[j][9]:  # 只要正确类不在最第一大概率的位置上了，就取第一大概率。第一大概率降到第二后，又被提升起来了。
                    y_second[j] = sorted_index[j][9]
            y_second = Variable(y_second.data, requires_grad=False)
            cost = criterion(h_adv, y_second)
        else:
            cost = -criterion(h_adv, y)

        net.zero_grad()
        if x_adv.grad is not None:  # 兜底
            x_adv.grad.data.fill_(0)
        cost.backward()

        x_adv.grad.sign_()
        grad = x_adv.grad
        x_adv = x_adv - alpha * grad  # x_adv.grad is not None:false  这一步后梯度清零了
        x_adv = where(x_adv > x + eps, x + eps, x_adv)
        x_adv = where(x_adv < x - eps, x - eps, x_adv)
        x_adv = torch.clamp(x_adv, x_val_min, x_val_max)  # 园本就有大于1的数据？因为是归一化到-1,1 之间了，所以0.999 加上0.1 也不能超过1，给clamp了
        x_adv = Variable(x_adv.data, requires_grad=True)  #  梯度会清零。 x_adv.data 取出来貌似就是requires——false的
        # 清零梯

    # h = net(x)
    # h_adv = net(x_adv)

    return x_adv







def taichi_fgsm_jiou(net, x, y, targeted_sxb=False, eps=1, alpha=1, iteration=1, x_val_min=-1, x_val_max=1,
           criterion=None,sxb=8):
    x_adv = Variable(x.data, requires_grad=True)  # 老版本的tensor 不可以求导，需要包装秤variable。新版本variable和tensor合并到一起了。
    # iteration = math.ceil((1.25*eps) / alpha)  # 把 eps 平均分配到每个iteration
    iteration = math.ceil(eps / alpha)  # 把 eps 平均分配到每个iteration
    # eps = format(eps / 255, '.2f')  # 转换成 小数形式
    eps = float(format(eps / 255, '.4f'))  # 转换成 小数形式
    alpha = alpha / 255 # 转换成 小数形式
    for i in range(iteration):
        h_adv = net(x_adv)
        if (i % 2) == 0:  # 目标为第二高的分类
            _, sorted_index = torch.sort(h_adv)
            # 向量加速todo
            y_second = sorted_index[:, sxb]
            for j in range(0, y.shape[0]):
                if y[j] != sorted_index[j][9]:  # 只要正确类不在最第一大概率的位置上了，就取第一大概率。第一大概率降到第二后，又被提升起来了。
                    y_second[j] = sorted_index[j][9]
            y_second = Variable(y_second.data, requires_grad=False)
            cost = criterion(h_adv, y_second)
        else:
            cost = -criterion(h_adv, y)

        net.zero_grad()  # 查看下 是否有梯度？ debug
        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)
        cost.backward()

        x_adv.grad.sign_()
        grad = x_adv.grad
        x_adv = x_adv - alpha * grad  # x_adv.grad is not None:false  这一步后梯度清零了
        # white = white - alpha * grad
        x_adv = where(x_adv > x + eps, x + eps, x_adv)
        x_adv = where(x_adv < x - eps, x - eps, x_adv)
        x_adv = torch.clamp(x_adv, x_val_min, x_val_max)  # 因为是归一化到-1,1 之间了，所以0.999 加上0.1 也不能超过1，给clamp了
        x_adv = Variable(x_adv.data, requires_grad=True)  # todo 清零？？  梯度会清零。 x_adv.data 取出来貌似就是requires——false的
        # todo清零梯度？？？


    return x_adv  # , white



def taichi_fgsm_half_half(net, x, y, targeted_sxb=False, eps=1, alpha=1, iteration=1, x_val_min=-1, x_val_max=1,
           criterion=None,sxb=8):
    x_adv = Variable(x.data, requires_grad=True)  # 老版本的tensor 不可以求导，需要包装秤variable。新版本variable和tensor合并到一起了。
    # iteration = math.ceil((1.25*eps) / alpha)  # 把 eps 平均分配到每个iteration
    iteration = math.ceil(eps / alpha)  # 把 eps 平均分配到每个iteration
    # eps = format(eps / 255, '.2f')  # 转换成 小数形式
    eps = float(format(eps / 255, '.4f'))  # 转换成 小数形式
    alpha = alpha / 255 # 转换成 小数形式

    for i in range(iteration):
        h_adv = net(x_adv)
        if i>=1:  # 目标为第二高的分类
            _, sorted_index = torch.sort(h_adv)
            # 向量加速todo
            y_second = sorted_index[:, sxb]
            for j in range(0, y.shape[0]):
                if y[j] != sorted_index[j][9]:  # 只要正确类不在最第一大概率的位置上了，就取第一大概率。第一大概率降到第二后，又被提升起来了。
                    y_second[j] = sorted_index[j][9]
            y_second = Variable(y_second.data, requires_grad=False)
            cost = criterion(h_adv, y_second)
        else:
            cost = -criterion(h_adv, y)

        net.zero_grad()  # 查看下 是否有梯度？ debug
        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)
        cost.backward()

        x_adv.grad.sign_()
        grad = x_adv.grad
        x_adv = x_adv - alpha * grad  # x_adv.grad is not None:false  这一步后梯度清零了
        # white = white - alpha * grad
        x_adv = where(x_adv > x + eps, x + eps, x_adv)
        x_adv = where(x_adv < x - eps, x - eps, x_adv)
        x_adv = torch.clamp(x_adv, x_val_min, x_val_max)  # 因为是归一化到-1,1 之间了，所以0.999 加上0.1 也不能超过1，给clamp了
        x_adv = Variable(x_adv.data, requires_grad=True)  # todo 清零？？  梯度会清零。 x_adv.data 取出来貌似就是requires——false的
        # todo清零梯度？？？

    # h = net(x)
    # h_adv = net(x_adv)

    return x_adv # , white


def taichi_fgsm_one_loss(net, x, y, targeted_sxb=False, eps=1, alpha=1, iteration=1, x_val_min=-1, x_val_max=1,
           criterion=None,sxb=8):
    x_adv = Variable(x.data, requires_grad=True)  # 老版本的tensor 不可以求导，需要包装秤variable。新版本variable和tensor合并到一起了。
    # iteration = math.ceil((1.25*eps) / alpha)  # 把 eps 平均分配到每个iteration
    iteration = math.ceil(eps / alpha)  # 把 eps 平均分配到每个iteration
    # eps = format(eps / 255, '.2f')  # 转换成 小数形式
    eps = float(format(eps / 255, '.4f'))  # 转换成 小数形式
    alpha = alpha / 255 # 转换成 小数形式


    for i in range(iteration):
        h_adv = net(x_adv)

        _, sorted_index = torch.sort(h_adv)
            # 向量加速todo
        y_second = sorted_index[:, sxb]
        for j in range(0, y.shape[0]):
            if y[j] != sorted_index[j][9]:  # 只要正确类不在最第一大概率的位置上了，就取第一大概率。第一大概率降到第二后，又被提升起来了。
                y_second[j] = sorted_index[j][9]
        y_second = Variable(y_second.data, requires_grad=False)
        cost1 = criterion(h_adv, y_second)

        cost2 = -criterion(h_adv, y)
        cost = 0.01*cost1+0.99*cost2
        net.zero_grad()  # 查看下 是否有梯度？ debug
        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)
        cost.backward()

        x_adv.grad.sign_()
        grad = x_adv.grad
        x_adv = x_adv - alpha * grad  # x_adv.grad is not None:false  这一步后梯度清零了
        # white = white - alpha * grad
        x_adv = where(x_adv > x + eps, x + eps, x_adv)
        x_adv = where(x_adv < x - eps, x - eps, x_adv)
        x_adv = torch.clamp(x_adv, x_val_min, x_val_max)  # 因为是归一化到-1,1 之间了，所以0.999 加上0.1 也不能超过1，给clamp了
        x_adv = Variable(x_adv.data, requires_grad=True)  # todo 清零？？  梯度会清零。 x_adv.data 取出来貌似就是requires——false的
        # todo清零梯度？？？

    # h = net(x)
    # h_adv = net(x_adv)

    return x_adv # , white








def sib_fgsm(net, x, y, targeted=False, eps=0.03, alpha=2 / 255, iteration=1, x_val_min=-1, x_val_max=1,
             criterion=None):
    return


def i_fgsm_white(net, x, y, targeted=False, eps=0.03, alpha=2 / 255, iteration=1, x_val_min=-1, x_val_max=1,
                 criterion=None):
    return

# num_iter = int(min(epsilon * 255 + 4, 1.25 * epsilon * 255))  # todo why
# adv_images = images
# transform_convert.transform_convert_save(images, transforms)  # todo #
# for i in range(num_iter):
#     outputs = model(adv_images)
#     loss = criterion(outputs, targets)
#     loss.backward()
#
#     # Generate perturbation todo 多余的部分剪裁？  归一化后的图片值是否是在-1,1之间
#     grad_j = torch.sign(adv_images.grad.data)
#     next_adv_images = adv_images + epsilon * grad_j
#     lower_adv_images = torch.max(torch.tensor(0.).to(device), torch.max(adv_images - epsilon, next_adv_images))
#     adv_images = torch.min(torch.tensor(1.).to(device), torch.min(adv_images + epsilon, lower_adv_images))
#     adv_images = Variable(adv_images.to(device), requires_grad=True)  # todo requires_grad=True ?
# test adversarial example or after gen all adversarial
#
#
# # because of image value [0, 1], args.epsilon * 255
# # number of iteration
# args.epsilon = 0.05
# device = 'cpu'
# num_iter = int(min(args.epsilon*255 +4, 1.25*args.epsilon*255))
#
# # X_0 adv images
# adv_images = images
# for i in range(num_iter):
#     outputs = model(adv_images)
#     loss = criterion(outputs, targets)
#     loss.backward()
#
#     # Generate perturbation
#     grad_j = torch.sign(adv_images.grad.data)
#     next_adv_images = adv_images + args.epsilon * grad_j
#     lower_adv_images = torch.max(torch.tensor(0.).to(device),torch.max(adv_images-args.epsilon, next_adv_images))
#     adv_images = torch.min(torch.tensor(1.).to(device), torch.min(adv_images+args.epsilon, lower_adv_images))
#     adv_images = Variable(adv_images.to(device), requires_grad=True)
#
# return num_iter, adv_images
