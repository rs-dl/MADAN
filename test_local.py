import torch.nn as nn
import os
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchvision import transforms
from dataset.data_loader import GetLoader
from torchvision import datasets


def test(dataset_name, epoch, domain, my_net_path, cla_net_path):
    assert dataset_name in ['0', '1', '3', '4']

    model_root = os.path.join('.', 'models', 'oilPalm', domain)
    image_root = os.path.join('.', 'dataset', dataset_name)

    cuda = True
    cudnn.benchmark = True
    batch_size = 128
    image_size = 17
    alpha = 0

    """load data"""

    img_transform_source = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    img_transform_target = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    
    test_list = os.path.join(image_root, '{}_test_labels.txt'.format(dataset_name))

    dataset = GetLoader(
        data_root=os.path.join(image_root, 'test'),
        data_list=test_list,
        transform=img_transform_target
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8
    )

    """ training """

    my_net = torch.load(my_net_path)
    my_net = my_net.eval()
    cla_net = torch.load(cla_net_path)
    cla_net = cla_net.eval()

    if cuda:
        my_net = my_net.cuda()
        cla_net = cla_net.cuda()

    len_dataloader = len(dataloader)
    data_target_iter = iter(dataloader)

    i = 0
    n_total = 0
    n_correct = 0

    while i < len_dataloader:

        # test model using target data
        data_target = data_target_iter.next()
        t_img, t_label = data_target

        batch_size = len(t_label)

        input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)
        class_label = torch.LongTensor(batch_size)

        if cuda:
            t_img = t_img.cuda()
            t_label = t_label.cuda()
            input_img = input_img.cuda()
            class_label = class_label.cuda()

        input_img.resize_as_(t_img).copy_(t_img)
        class_label.resize_as_(t_label).copy_(t_label)

        feature = my_net(input_data=input_img, alpha=alpha)
        class_output = cla_net(feature)
        class_output_softmax = nn.Softmax(dim=1)(class_output)
        pred = class_output.data.max(1, keepdim=True)[1]
        n_correct += pred.eq(class_label.data.view_as(pred)).cpu().sum()
        n_total += batch_size

        i += 1

    accu = n_correct.data.numpy() * 1.0 / n_total

    print('epoch: %d, accuracy of the %s dataset: %f' % (epoch, dataset_name, accu))
    return accu
