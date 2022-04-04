import torch
import numpy as np

x = torch.tensor([[[[1., 1., 1., 1.],
                [2., 2., 2., 2.],
                [3., 3., 3., 3.],
                [4., 4., 4., 4.]],
               [[1., 1., 1., 1.],
                [1., 1., 1., 1.],
                [3., 3., 3., 3.],
                [3., 3., 3., 3.]],
               [[2., 2., 2., 2.],
                [2., 2., 2., 2.],
                [4., 4., 4., 4.],
                [4., 4., 4., 4.]]],
              [[[0., 0., 0., 0.],
                [2., 2., 2., 2.],
                [3., 3., 3., 3.],
                [6., 6., 6., 6.]],
               [[1., 1., 1., 1.],
                [5., 5., 5., 5.],
                [3., 3., 3., 3.],
                [3., 3., 3., 3.]],
               [[2., 2., 2., 2.],
                [2., 2., 2., 2.],
                [4., 4., 4., 4.],
                [4., 4., 4., 4.]]]])

x_hat = torch.tensor([[[[1., 1., 1., 1.],
                [2., 2., 2., 2.],
                [3., 3., 3., 3.],
                [4., 4., 4., 4.]],
               [[1., 1., 1., 1.],
                [1., 1., 1., 1.],
                [3., 3., 3., 3.],
                [3., 3., 7., 3.]],
               [[2., 2., 2., 2.],
                [2., 2., 2., 2.],
                [4., 4., 4., 4.],
                [4., 4., 4., 4.]]],
              [[[0., 0., 0., 0.],
                [2., 2., 2., 2.],
                [3., 3., 3., 3.],
                [6., 6., 6., 6.]],
               [[1., 0., 1., 1.],
                [5., 5., 5., 9.],
                [3., 3., 3., 3.],
                [3., 3., 3., 3.]],
               [[2., 2., 2., 2.],
                [2., 2., 2., 2.],
                [4., 4., 4., 4.],
                [2., 4., 4., 4.]]]])

print(np.array(x.size()))

loss_list = []
for i in range(np.array(x.size())[0]):
    loss = torch.sqrt((x_hat[i, :, :, :] - x[i, :, :, :]).pow(2).mean())
    loss_list.append(loss.item())
print(torch.sqrt((x_hat - x).pow(2).mean()))
listed = []
for i in range(2):
    listed.append(torch.sqrt((x_hat[i, :, :, :] - x[i, :, :, :]).pow(2).mean()))
print(sum(listed)/len(listed))

# 내림차순 정렬
loss_list.sort(reverse=True)
print(loss_list)

loss_list = [0.1, 0.2, 0.3]
list = []

for i in range(3):
    list.append(loss_list[i]*(1/loss_list[-(i+1)]))
print(list)
print(sum(list)/len(list))

print(torch.Tensor(list))
print(torch.Tensor())

print(torch.sqrt((x_hat - x).pow(2).mean()))
print(torch.tensor(333))

print(np.array(x).mean())