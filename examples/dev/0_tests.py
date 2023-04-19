import torch

torch.cuda.is_available()

"""That's great, let us import then a few libraries, which we'll be using during this tutorial!"""

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from examples.dev.continualai.colab.scripts import mnist

mnist.init()

x_train, t_train, x_test, t_test = mnist.load()

print("x_train dim and type: ", x_train.shape, x_train.dtype)
print("t_train dim and type: ", t_train.shape, t_train.dtype)
print("x_test dim and type: ", x_test.shape, x_test.dtype)
print("t_test dim and type: ", t_test.shape, t_test.dtype)

"""Let's take a look at the actual images!"""

# f, axarr = plt.subplots(2, 2)
# axarr[0, 0].imshow(x_train[1, 0], cmap="gray")
# axarr[0, 1].imshow(x_train[2, 0], cmap="gray")
# axarr[1, 0].imshow(x_train[3, 0], cmap="gray")
# axarr[1, 1].imshow(x_train[4, 0], cmap="gray")
# np.vectorize(lambda ax: ax.axis('off'))(axarr);

"""Good! Let's now set up a few general setting before using torch..."""

# switch to False to use CPU
use_cuda = True

use_cuda = use_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(1)

"""... and define our first conv-net! We will use 3 layers of convolutions and two fully connected layers:"""


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


"""Then we can write the *train* and *test* functions. Note that for simplicity here we are not using PyTorch [Data Loaders](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html) but this is not recommended for efficiency."""


def train(model, device, x_train, t_train, optimizer, epoch):
    model.train()

    for start in range(0, len(t_train) - 1, 256):
        end = start + 256
        x, y = torch.from_numpy(x_train[start:end]), torch.from_numpy(t_train[start:end]).long()
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        output = model(x)
        loss = F.cross_entropy(output, y)
        loss.backward()
        optimizer.step()
        # print(loss.item())
    print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, loss.item()))


def test(model, device, x_test, t_test):
    model.eval()
    test_loss = 0
    correct = 0
    for start in range(0, len(t_test) - 1, 256):
        end = start + 256
        with torch.no_grad():
            x, y = torch.from_numpy(x_test[start:end]), torch.from_numpy(t_test[start:end]).long()
            x, y = x.to(device), y.to(device)
            output = model(x)
            test_loss += F.cross_entropy(output, y).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max logit
            correct += pred.eq(y.view_as(pred)).sum().item()

    test_loss /= len(t_test)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(t_test),
        100. * correct / len(t_test)))
    return 100. * correct / len(t_test)


"""Then we are ready to instantiate our model and start the training!"""

# model = Net().to(device)
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
#
# for epoch in range(1, 3):
#     train(model, device, x_train, t_train, optimizer, epoch)
#     test(model, device, x_test, t_test)

"""Wow! 94% accuracy in such a short time. 

**Questions to explore:**

*   Can you find a better parametrization to improve the final accuracy?
*   Can you change the network architecture to improve the final accuracy?
*   Can you achieve the same performances with a smaller architecture?
*   What's the difference in accuracy if you change convolutions with fully connected layers?

Some tips here: http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#4d4e495354

But what if now we want we the same model being able to solve a new task we encounter over time like a permuted version of the same MNIST? Let's define our custom function to permute it!
"""


def permute_mnist(mnist, seed):
    """ Given the training set, permute pixels of each img the same way. """

    np.random.seed(seed)
    print("starting permutation...")
    h = w = 28
    perm_inds = list(range(h * w))
    np.random.shuffle(perm_inds)
    # print(perm_inds)
    perm_mnist = []
    for set in mnist:
        num_img = set.shape[0]
        flat_set = set.reshape(num_img, w * h)
        perm_mnist.append(flat_set[:, perm_inds].reshape(num_img, 1, w, h))
    print("done.")
    return perm_mnist


# x_train2, x_test2 = permute_mnist([x_train, x_test], 0)

# f, axarr = plt.subplots(1, 2)
# axarr[0].imshow(x_train[1, 0], cmap="gray")
# axarr[1].imshow(x_train2[2, 0], cmap="gray")
# np.vectorize(lambda ax: ax.axis('off'))(axarr);

"""Amazing. Now let's see how our pre-trained model is working on both the original and the permuted MNIST dataset:"""

# print("Testing on the first task:")
# test(model, device, x_test, t_test)
#
# print("Testing on the second task:")
# test(model, device, x_test2, t_test);
#
# """Mmmh... that's pretty bad, our model cannot generalize to this apparently very different new task! Well, we can just finetune our model using the new permuted training set!"""
#
# for epoch in range(1, 3):
#     train(model, device, x_train2, t_train, optimizer, epoch)
#     test(model, device, x_test2, t_test)
#
# print("Testing on the first task:")
# test(model, device, x_test, t_test)
#
# print("Testing on the second task:")
# test(model, device, x_test2, t_test);

"""This is very annoying! Now we are not able to solve the original MNIST task anymore! :-( This is the phenomenon known in literature as **Catastrophic Forgetting**! In the following section we well compare three different strategies for learning continually (and trying to not forget!)

**Questions to explore:**

*   When the permuted MNIST benchmark has been firstly introduced? 
*   Can simple Dropout and Regularization techniques reduce forgetting?
*   In the permuted MNIST task, do convolutions still help increasing the accuracy?

Some tips here: https://papers.nips.cc/paper/5059-compete-to-compute

## CL Strategies

Let us now focus on some strategies for reducing catastrofic forgetting, one of the principal problems when learning continuously. in this section we will take a look at three different strategies:

1.   Naive
2.   Rehearsal
3.   Elastic Weight Consolidation (EWC)

and run it on a 3-tasks Permuted MNIST. Finally we will plot our results for comparison. For a more comprehensive overview on recent CL strategies for deep learning take a look at the recent paper "[Continuous Learning in Single-Incremental-Task Scenarios](https://arxiv.org/abs/1806.08568)".

Let's start by defining our 3 tasks with the function we have already introduced before:
"""

# task 1
task_1 = [(x_train, t_train), (x_test, t_test)]

# task 2
x_train2, x_test2 = permute_mnist([x_train, x_test], 1)
task_2 = [(x_train2, t_train), (x_test2, t_test)]

# task 3
x_train3, x_test3 = permute_mnist([x_train, x_test], 2)
task_3 = [(x_train3, t_train), (x_test3, t_test)]

# task list
tasks = [task_1, task_2, task_3]

"""### Naive Strategy

The  *Naive* strategy, is the simple idea of continuing the back-prop process on the new batches/tasks. This is very simple, but at the same time very prone to forgetting as we have witnessed before. Let's how it works on three tasks:
"""

# model = Net().to(device)
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
#
# naive_accs = []
#
# for id, task in enumerate(tasks):
#     avg_acc = 0
#     print("Training on task: ", id)
#
#     (x_train, t_train), _ = task
#
#     for epoch in range(1, 2):
#         train(model, device, x_train, t_train, optimizer, epoch)
#
#     for id_test, task in enumerate(tasks):
#         print("Testing on task: ", id_test)
#         _, (x_test, t_test) = task
#         acc = test(model, device, x_test, t_test)
#         avg_acc = avg_acc + acc
#
#     naive_accs.append(avg_acc / 3)
#     print("Avg acc: ", avg_acc / 3)

"""**Questions to explore:**

*   Does the order of the tasks effect the final results? 

Some tips here: http://proceedings.mlr.press/v78/lomonaco17a/lomonaco17a.pdf

### Rehearsal Strategy

Another simple CL idea is to carry on *all* or *part* of the previously encountered examples (of the previous tasks), shuffling them with the data of the current task. Using *all* the past data is near to the optimal performance we can desire at the end of the task sequence but at the expense of much bigger memory usage.

Let's start by defining a function to shuffle our data:
"""


def shuffle_in_unison(dataset, seed, in_place=False):
    """ Shuffle two (or more) list in unison. """

    np.random.seed(seed)
    rng_state = np.random.get_state()
    new_dataset = []
    for x in dataset:
        if in_place:
            np.random.shuffle(x)
        else:
            new_dataset.append(np.random.permutation(x))
        np.random.set_state(rng_state)

    if not in_place:
        return new_dataset


"""Now we can reset the model and optimizer and run our training over the tasks sequence:"""

# model = Net().to(device)
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
#
# rehe_accs = []
# for id, task in enumerate(tasks):
#     avg_acc = 0
#     print("Training on task: ", id)
#
#     (x_train, t_train), _ = task
#
#     # for previous task
#     for i in range(id):
#         (past_x_train, past_t_train), _ = tasks[i]
#         x_train = np.concatenate((x_train, past_x_train))
#         t_train = np.concatenate((t_train, past_t_train))
#
#     x_train, t_train = shuffle_in_unison([x_train, t_train], 0)
#
#     for epoch in range(1, 2):
#         train(model, device, x_train, t_train, optimizer, epoch)
#
#     for id_test, task in enumerate(tasks):
#         print("Testing on task: ", id_test)
#         _, (x_test, t_test) = task
#         acc = test(model, device, x_test, t_test)
#         avg_acc = avg_acc + acc
#
#     print("Avg acc: ", avg_acc / 3)
#     rehe_accs.append(avg_acc / 3)

"""**Questions to explore:**

*   Can you find a way to reduce the number of examples of the previous tasks to maintain in memory? 
*   Can you find a good trade-off between memory overhead and final accuracy?
*   Why is shuffling needed here?

Some tips here: https://arxiv.org/abs/1809.05922

### Elastic Weights Consolidation (EWC) Strategy

Elastic Weights Consolidation (EWC) is a common CL strategy firstly proposed in the paper: "[Overcoming catastrophic forgetting in neural networks](https://arxiv.org/abs/1612.00796)" for deep neural networks.

It is based on the computation of the importance of each weight (fisher information) and a squared regularization loss, penalizing changes in the most important wheights for the previous tasks.

It has the great advantage of **not using any** of the previous tasks data!
"""

fisher_dict = {}
optpar_dict = {}
ewc_lambda = 0.4

model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

"""Now we need to define an additional function to compute the fisher information for each weight at the end of each task:"""


def on_task_update(task_id, x_mem, t_mem):
    model.train()
    optimizer.zero_grad()

    # accumulating gradients
    for start in range(0, len(t_mem) - 1, 256):
        end = start + 256
        x, y = torch.from_numpy(x_mem[start:end]), torch.from_numpy(t_mem[start:end]).long()
        x, y = x.to(device), y.to(device)
        output = model(x)
        loss = F.cross_entropy(output, y)
        loss.backward()

    fisher_dict[task_id] = {}
    optpar_dict[task_id] = {}

    # gradients accumulated can be used to calculate fisher
    for name, param in model.named_parameters():
        optpar_dict[task_id][name] = param.data.clone()
        fisher_dict[task_id][name] = param.grad.data.clone().pow(2)


"""We need also to modify our *train* function to add the new regularization loss:"""


def train_ewc(model, device, task_id, x_train, t_train, optimizer, epoch):
    model.train()

    for start in range(0, len(t_train) - 1, 256):  # One epoch. Batch size = 256
        end = start + 256
        x, y = torch.from_numpy(x_train[start:end]), torch.from_numpy(t_train[start:end]).long()
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        output = model(x)
        loss = F.cross_entropy(output, y)

        ### magic here! :-)
        for task in range(task_id):  # If task is 0, skip
            for name, param in model.named_parameters():
                fisher = fisher_dict[task][name]
                optpar = optpar_dict[task][name]
                loss += (fisher * (optpar - param).pow(2)).sum() * ewc_lambda

        loss.backward()
        optimizer.step()
        # print(loss.item())
    print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, loss.item()))


"""Finally we can run the train over the three tasks sequence of th *Permuted MNIST*:"""

ewc_accs = []
for id, task in enumerate(tasks):
    avg_acc = 0
    print("Training on task: ", id)

    (x_train, t_train), _ = task

    for epoch in range(1, 3):
        train_ewc(model, device, id, x_train, t_train, optimizer, epoch)
    on_task_update(id, x_train, t_train)

    for id_test, task in enumerate(tasks):
        print("Testing on task: ", id_test)
        _, (x_test, t_test) = task
        acc = test(model, device, x_test, t_test)
        avg_acc = avg_acc + acc

    print("Avg acc: ", avg_acc / 3)
    ewc_accs.append(avg_acc / 3)

"""**Questions to explore:**

*   How much the `ewc_lambda` parameter effect the final results? 
*   Can you find a better parametrization to improve stability?
*   Can you find the memory overhead introduced by EWC with respect to the Naive approach?

Some tips here: https://arxiv.org/pdf/1805.06370.pdf

### Plot Results

To conclude, let's summerize our results in a nice plot! :-)
"""

# plt.plot([1, 2, 3], naive_accs, '-o', label="Naive")
# plt.plot([1, 2, 3], rehe_accs, '-o', label="Rehearsal")
# plt.plot([1, 2, 3], ewc_accs, '-o', label="EWC")
# plt.xlabel('Tasks Encountered', fontsize=14)
# plt.ylabel('Average Accuracy', fontsize=14)
# plt.title('CL Strategies Comparison on MNIST', fontsize=14);
# plt.xticks([1, 2, 3])
# plt.legend(prop={'size': 16});

"""**Questions to explore:**

*   What's the difference in terms of memory utilization among the three methods? 
*   Can you plot a similar graph highlighting the memory increase over time?

Some tips here: https://stackoverflow.com/questions/449560/how-do-i-determine-the-size-of-an-object-in-python/30316760

**Copyright (c) 2018. Continual AI. All rights reserved. **

See the accompanying LICENSE file in the GitHub repository for terms. 

*Date: 29-09-2018                                                             
Author: Vincenzo Lomonaco                                                    
E-mail: contact@continualai.org                                           
Website: continualai.org*
"""