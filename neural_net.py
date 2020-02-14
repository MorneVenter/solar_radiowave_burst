import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


img_path = 'output.png'
bgr_img = cv2.imread(img_path)
gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
gray_img = gray_img.astype("float32")/255

filter_vals = np.array([[-1, -1, 1, 1], [-1, -1, 1, 1], [-1, -1, 1, 1], [-1, -1, 1, 1]])
filter_1 = filter_vals
filter_2 = -filter_1
filter_3 = filter_1.T
filter_4 = -filter_3
filters = np.array([filter_1, filter_2, filter_3, filter_4])



# Neural network with one convolutional layer with four filters
class Net(nn.Module):
    def __init__(self, weight):
        super(Net, self).__init__()
        k_height, k_width = weight.shape[2:]
        self.conv = nn.Conv2d(1, 4, kernel_size=(k_height, k_width), bias=False)
        self.conv.weight = torch.nn.Parameter(weight)
    def forward(self, x):
        conv_x = self.conv(x)
        activated_x = F.relu(conv_x)
        return conv_x, activated_x


weight = torch.from_numpy(filters).unsqueeze(1).type(torch.FloatTensor)
model = Net(weight)

# def viz_layer(layer, n_filters = 4):
#     fig = plt.figure(figsize=(20, 20))
#     for i in range(n_filters):
#         ax = fig.add_subplot(1, n_filters, i+1, xticks=[], yticks=[])
#         ax.imshow(np.squeeze(layer[0,i].data.numpy()), cmap='gray')
#         ax.set_title('Output %s' % str(i+1))
#     plt.show()

# plt.imshow(gray_img, cmap='gray')
# fig = plt.figure(figsize=(12, 6))
# fig.subplots_adjust(left=0, right=1.5, bottom=0.8, top=1, hspace=0.05, wspace=0.05)
# for i in range(4):
#     ax = fig.add_subplot(1, 4, i+1, xticks=[], yticks=[])
#     ax.imshow(filters[i], cmap='gray')
#     ax.set_title('Filter %s' % str(i+1))

gray_img_tensor = torch.from_numpy(gray_img).unsqueeze(0).unsqueeze(1)
conv_layer, activated_layer = model(gray_img_tensor)

gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
# 3x3 array for edge detection
sobel_y = np.array([[ -1, -2, -1],
                   [ 0, 0, 0],
                   [ 1, 2, 1]])
sobel_x = np.array([[ -1, 0, 1],
                   [ 0, 0, 0],
                   [ 1, 2, 1]])

filtered_image = cv2.filter2D(gray, -1, sobel_y)
