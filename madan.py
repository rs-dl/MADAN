import random
import sys
import torch.nn as nn
import os
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from dataset.data_loader import GetLoader
from torchvision import datasets
from torchvision import transforms
from models.model_5b import CNNModel, Adver, Classifier
import numpy as np
from test_local import test

source_dataset_name = sys.argv[1]
target_dataset_name = sys.argv[2]
saveName = "CLGE_5b"
resultsFile = os.path.join('.', 'valRes', 'oilPalm', '{}-{}'.format(source_dataset_name, target_dataset_name), '{}.txt'.format(saveName))
res = open(resultsFile, 'w')
source_image_root = os.path.join('.', 'dataset', source_dataset_name)
target_image_root = os.path.join('.', 'dataset', target_dataset_name)
model_root = os.path.join('.', 'models', 'oilPalm', '{}-{}'.format(source_dataset_name, target_dataset_name))
cuda = True
cudnn.benchmark = True
lr = 1e-3
batch_size = 128
image_size = 17
n_epoch = 100
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[3]

manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)

# load data

img_transform_source = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

img_transform_target = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

source_list = os.path.join(source_image_root, '{}_train_labels.txt'.format(source_dataset_name))

dataset_source = GetLoader(
    data_root=os.path.join(source_image_root, 'train'),
    data_list=source_list,
    transform=img_transform_source
)

dataloader_source = torch.utils.data.DataLoader(
    dataset=dataset_source,
    batch_size=batch_size,
    shuffle=True)

target_list = os.path.join(target_image_root, '{}_train_labels.txt'.format(target_dataset_name))

dataset_target = GetLoader(
    data_root=os.path.join(target_image_root, 'train'),
    data_list=target_list,
    transform=img_transform_target
)

dataloader_target = torch.utils.data.DataLoader(
    dataset=dataset_target,
    batch_size=batch_size,
    shuffle=True)

# load model
if source_dataset_name == "4" or source_dataset_name == "3":
    output = 3
else:
    output = 4

my_net = CNNModel()
ad_net = Adver()
cla_net = Classifier(output_num=output)

# setup optimizer

optimizer = optim.Adam(my_net.parameters(), lr=lr)

loss_class = torch.nn.CrossEntropyLoss()
# loss_domain = torch.nn.NLLLoss()
loss_domain = torch.nn.BCELoss()

if cuda:
    my_net = my_net.cuda()
    ad_net = ad_net.cuda()
    cla_net = cla_net.cuda()
    loss_class = loss_class.cuda()
    loss_domain = loss_domain.cuda()

for p in my_net.parameters():
    p.requires_grad = True


# training
def Entropy(x, attention_global):
    mask = x.ge(0.000001)
    # print mask.cpu().data.numpy()
    mask_out = torch.masked_select(x, mask)
    attention_global_out = torch.masked_select(attention_global, mask)
    # print mask_out.cpu().data.numpy()
    entropy = -(torch.sum(attention_global_out.data * mask_out * torch.log(mask_out)))
    # print entropy.cpu().data.numpy()
    return entropy / float(x.size(0))


def H(x):
    return -x * torch.log2(x + 1e-10) - (1 - x) * torch.log2(1 - x + 1e-10)


for epoch in range(n_epoch):

    len_dataloader = min(len(dataloader_source), len(dataloader_target))
    data_source_iter = iter(dataloader_source)
    data_target_iter = iter(dataloader_target)

    i = 0
    while i < len_dataloader:

        p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # training model using source data
        data_source = data_source_iter.next()
        s_img, s_label = data_source
        my_net.zero_grad()
        batch_size = len(s_label)
        # print(len(s_label))
        one_tensor = torch.ones(batch_size, 1)
        input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)
        class_label = torch.LongTensor(batch_size)
        domain_label = torch.zeros(batch_size)
        domain_label = domain_label.long()
        # domain_label = torch.from_numpy(np.array([[0]] * batch_size)).float()
        if cuda:
            s_img = s_img.cuda()
            s_label = s_label.cuda()
            input_img = input_img.cuda()
            class_label = class_label.cuda()
            domain_label = domain_label.cuda()
            one_tensor = one_tensor.cuda()
        input_img.resize_as_(s_img).copy_(s_img)
        class_label.resize_as_(s_label).copy_(s_label)
        # print(input_img.size())
        # class_output_source, domain_output = my_net(input_data=input_img, alpha=alpha)
        s_feature = my_net(input_data=input_img, alpha=alpha)
        s_local_probability = ad_net(s_feature, alpha)
        s_local_out = H(s_local_probability)
        s_global_feature = (one_tensor.data + s_local_out.data) * s_feature
        
        s_class_output = cla_net(s_global_feature)
        s_class_output_softmax = nn.Softmax(dim=1)(s_class_output)
        err_s_label = loss_class(s_class_output, class_label)
        err_s_local = loss_domain(s_local_probability.float().view(-1), domain_label.float().view(-1))

        s_global_probability = ad_net(s_global_feature, alpha)
        err_s_global = loss_domain(s_global_probability.float().view(-1), domain_label.float().view(-1))
        s_global_out = H(s_global_probability)
        s_global_attention = 1 + s_global_out.data
        err_s_entropy = Entropy(s_class_output_softmax, s_global_attention)

        # training model using target data
        data_target = data_target_iter.next()
        t_img, _ = data_target

        batch_size = len(t_img)

        input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)
        domain_label = torch.ones(batch_size)
        domain_label = domain_label.long()
        one_tensor = torch.ones(batch_size, 1)

        if cuda:
            t_img = t_img.cuda()
            input_img = input_img.cuda()
            domain_label = domain_label.cuda()
            one_tensor = one_tensor.cuda()

        input_img.resize_as_(t_img).copy_(t_img)

        t_feature = my_net(input_data=input_img, alpha=alpha)
        t_local_probability = ad_net(t_feature, alpha)
        t_local_out = H(t_local_probability)
        t_global_feature = (one_tensor.data + t_local_out.data) * t_feature
        t_class_output = cla_net(t_global_feature)
        t_class_output_softmax = nn.Softmax(dim=1)(t_class_output)
        
        err_t_local = loss_domain(t_local_probability.float().view(-1), domain_label.float().view(-1))
        
        t_global_probability = ad_net(t_global_feature, alpha)
        err_t_global = loss_domain(t_global_probability.float().view(-1), domain_label.float().view(-1))
        t_global_out = H(t_global_probability)
        t_global_attention = 1 + t_global_out.data
        err_t_entropy = Entropy(t_class_output_softmax, t_global_attention)
        err = (err_t_local + err_s_local) * 0.1 + err_s_label + (err_t_entropy + err_s_entropy) * 1 + (err_s_global + err_t_global) * 0.1
        err.backward()
        optimizer.step()

        i += 1

        print('epoch: %d, [iter: %d / all %d], err: %f, err_s_label: %f, err_s_local: %f, err_t_local: %f, err_s_entropy: %f, err_t_entropy: %f, err_s_global: %f, err_t_global: %f' % (epoch, i, len_dataloader, err.cpu().data.numpy(), err_s_label.cpu().data.numpy(), err_s_local.cpu().data.numpy(), err_t_local.cpu().data.numpy(), err_s_entropy.cpu().data.numpy(), err_t_entropy.cpu().data.numpy(), err_s_global.cpu().data.numpy(), err_t_global.cpu().data.numpy()))
    if (epoch + 1) % 10 == 0:
        torch.save(my_net, '{}/{}-fea-{}.pth'.format(model_root, saveName, epoch + 1))
        torch.save(cla_net, '{}/{}-cla-{}.pth'.format(model_root, saveName, epoch + 1))
        sourceAcc = test(source_dataset_name, epoch, '{}-{}'.format(source_dataset_name, target_dataset_name), '{}/{}-fea-{}.pth'.format(model_root, saveName, epoch + 1), '{}/{}-cla-{}.pth'.format(model_root, saveName, epoch + 1))
        targetAcc = test(target_dataset_name, epoch, '{}-{}'.format(source_dataset_name, target_dataset_name), '{}/{}-fea-{}.pth'.format(model_root, saveName, epoch + 1), '{}/{}-cla-{}.pth'.format(model_root, saveName, epoch + 1))
        res.write('source acc: {}, target acc: {}\n'.format(sourceAcc, targetAcc))
print('done')
