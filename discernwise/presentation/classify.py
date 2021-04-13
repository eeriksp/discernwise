# #!/usr/bin/env python
#
# import os
#
# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# from torch import nn
# from torch import optim
# import torch.nn.functional as F
# from torch.autograd import Variable
# from torchvision import datasets, transforms, models
#
# data_dir = os.path.join('.', 'data', 'test')
# test_transforms = transforms.Compose([transforms.Resize([224, 224]),
#                                       transforms.ToTensor(),
#                                       ])
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = torch.load('neuralnetwork.pth')
# model.eval()
#
#
# def predict_image(image):
#     image_tensor = test_transforms(image).float()
#     image_tensor = image_tensor.unsqueeze_(0)
#     input = Variable(image_tensor)
#     input = input.to(device)
#     output = model(input)
#     index = output.data.cpu().numpy().argmax()
#     return index
#
#
# def get_random_images(num):
#     data = datasets.ImageFolder(data_dir, transform=test_transforms)
#     global classes
#     classes = data.classes
#     indices = list(range(len(data)))
#     np.random.shuffle(indices)
#     idx = indices[:num]
#     from torch.utils.data.sampler import SubsetRandomSampler
#     sampler = SubsetRandomSampler(idx)
#     loader = torch.utils.data.DataLoader(data,
#                                          sampler=sampler, batch_size=num)
#     dataiter = iter(loader)
#     images, labels = dataiter.next()
#     return images, labels
#
# to_pil = transforms.ToPILImage()
# images, labels = get_random_images(10)
# print('labels: ', labels)
# fig = plt.figure(figsize=(10, 10))
# print(classes)
# for ii in range(len(images)):
#     image = to_pil(images[ii])
#     index = predict_image(image)
#     sub = fig.add_subplot(1, len(images), ii + 1)
#     res = int(labels[ii]) == index
#     sub.set_title(str(classes[index]) + ":" + str(res))
#     plt.axis('off')
#     plt.imshow(image)
# plt.show()
