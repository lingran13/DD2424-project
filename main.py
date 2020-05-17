import torch
import torchvision
import torchvision.transforms as transforms
from ShuffleNet import ShuffleNet
import torch.nn as nn
import torch.nn.functional as F
import time


batch_size = 128
epochs = 100
best_acc = 0
lr = 0.1

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False, num_workers=0)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

net = ShuffleNet(groups=8, in_channels=3, num_classes=10, scale_factor=0.5)
net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

start_time = time.time()

for epoch in range(epochs): # loop over the dataset multiple times
    
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        
        # print statistics
        running_loss += loss.item()
        # every 100 iteration, print loss
        if (i+1) % 100 == 0: 
            print('[%d, %5d] loss: %.3f' %
            (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

    scheduler.step()        

    print("Test!")
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %.1f %%' % (
    100 * correct / total))
    acc = 100 * correct / total
        
    # 记录最佳测试分类准确率并写入best_acc.txt文件中
    if acc > best_acc:
        f3 = open("best_acc.txt", "w")
        f3.write("epoch=%d,best_acc= %.1f%%" % (epoch + 1, acc))
        f3.close()
        best_acc = acc

print("Training Finished")

end_time = time.time()
print("total time %.2f min" %((end_time - start_time)/60))
print("total time %.1f h" %((end_time - start_time)/3600))

# save model
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)
