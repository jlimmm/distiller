import distiller
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from distiller.data_loggers import collector_context

num_epoch = 0
batch_size = 4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#####################################
# Helper functions
#####################################
def save_model_structure(path, filename, model):
    with open(path+filename, 'w') as text_file:
        text_file.write(str(model))
    return

def generate_yaml(model, model_name):
    distiller.utils.assign_layer_fq_names(model)
    #msglogger.info("Generating quantization calibration stats based on {0} users".format(args.qe_calibration))
    collector = distiller.data_loggers.QuantCalibrationStatsCollector(model)
    with collector_context(collector):
        # Here call your model evaluation function, making sure to execute only
        # the portion of the dataset specified by the qe_calibration argument
        pass
    path_yaml = './stat_yaml/' + model_name + '.yaml'
    collector.save(path_yaml)
    
    return

def generate_model_info(model, model_name):
    print('- %s, type:%s' %(model_name, type(model)))
    
    path_structure = './model_structure/'
    save_model_structure(path_structure, model_name+'.txt', model)

    return



########################################################################
# 1. Load and normalizing the CIFAR10 training and test datasets using
# torchvision
########################################################################

import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


path_c10 = '../../../data.cifar10/'
trainset = torchvision.datasets.CIFAR10(root=path_c10, train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root=path_c10, train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

########################################################################
# 2. Define a network
########################################################################

from distiller.models import create_model
import distiller.quantization as quant
from copy import deepcopy

model= create_model(pretrained=True, dataset='imagenet', arch='resnet18', parallel=False)
generate_model_info(model, 'dist_base')

model.to(device)

########################################################################
# 3. Difine a loss function and optimizer
########################################################################

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


########################################################################
# 4. Train the network
########################################################################

import time
start = time.time()

for epoch in range(num_epoch):  # loop over the dataset multiple times
    
    running_loss = 0.0
    model.train()
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            end = time.time()
            print('[%d, %5d] loss: %.3f, time:%.1f' %
                  (epoch + 1, i + 1, running_loss / 2000, end-start))
            running_loss = 0.0
            start = time.time()
                  
    val_loss = 0.0
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            val_loss += loss

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
                
        val_loss /= len(testloader) 
        
        print('\t val_loss: %.3f, acc:%.2f%%' %
                (val_loss, 100. * correct / total))

    

print('Finished Training')


########################################################################
# 5. generate stats
########################################################################
model_name = 'acts_quantization_stats.yaml'
path_yaml = './stat_yaml/' 
#path_yaml = './stat_yaml/' + model_name + '.yaml'
#generate_yaml(model, 'test_model')

# CHECK: /examples/word_language_model/quantize_lstm.ipynb
def evaluate(model):
    val_loss = 0.0
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            val_loss += loss

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
                
        val_loss /= len(testloader) 
    
    eval_acc = 100 * correct / total
    
    return val_loss, eval_acc

eval_loss, eval_acc = evaluate(model)
print('[eval] loss:%.3f, acc:%.2f%%' % (eval_loss, eval_acc))

def test_fn(model):
    return evaluate(model)[0]
        
from distiller.data_loggers import collect_quant_stats
collect_quant_stats(model, test_fn, save_dir=path_yaml) 


#####################################
# Quantization
#####################################
#model = model_not_parallel

quant_mode = {'activations': 'ASYMMETRIC_UNSIGNED', 'weights': 'SYMMETRIC'}
stats_file = path_yaml + model_name 
#stats_file = "../quantization/post_train_quant/stats/resnet18_quant_stats.yaml"
dummy_input = distiller.get_dummy_input(input_shape=model.input_shape)

quantizer = quant.PostTrainLinearQuantizer(
    deepcopy(model), bits_activations=8, bits_parameters=8, mode=quant_mode,
    model_activation_stats=stats_file, overrides=None
)
quantizer.prepare_model(dummy_input)

pyt_model = quantizer.convert_to_pytorch(dummy_input)
generate_model_info(pyt_model, 'after_quant')

print('Distiller model device:', distiller.model_device(quantizer.model))
print('PyTorch model device:', distiller.model_device(pyt_model))

print(pyt_model.layer1[0].conv1.weight().int_repr().data[0, 0, :, :])
print(pyt_model.layer1[0].conv1.weight().dequantize().data[0, 0, :, :])

#eval_loss, eval_acc = evaluate(pyt_model)
#print('[eval] loss:%.3f, acc:%.2f%%' % (eval_loss, eval_acc))


print('DISTILLER1:\n{}\n'.format(quantizer.model.conv1))
#print('DISTILLER2:\n{}\n'.format(quantizer.model.module.conv1))
print('PyTorch:\n{}\n'.format(pyt_model.conv1))

print('layer1.0.conv1')
print(pyt_model.layer1[0].conv1)
print('\nlayer1.0.add')
print(pyt_model.layer1[0].add)

import torchnet as tnt

def eval_model(data_loader, model, device, print_freq=10):
    print('Evaluating model')
    criterion = torch.nn.CrossEntropyLoss().to(device)

    loss = tnt.meter.AverageValueMeter()
    classerr = tnt.meter.ClassErrorMeter(accuracy=True, topk=(1, 5))

    total_samples = len(data_loader.sampler)
    batch_size = data_loader.batch_size
    total_steps = math.ceil(total_samples / batch_size)
    print('{0} samples ({1} per mini-batch)'.format(total_samples, batch_size))

    # Switch to evaluation mode
    model.eval()

    for step, (inputs, target) in enumerate(data_loader):
        with torch.no_grad():
            inputs, target = inputs.to(device), target.to(device)
            # compute output from model
            output = model(inputs)

            # compute loss and measure accuracy
            loss.add(criterion(output, target).item())
            classerr.add(output.data, target)

            if (step + 1) % print_freq == 0:
                print('[{:3d}/{:3d}] Top1: {:.3f}  Top5: {:.3f}  Loss: {:.3f}'.format(
                      step + 1, total_steps, classerr.value(1), classerr.value(5), loss.mean), flush=True)
    print('----------')
    print('Overall ==> Top1: {:.3f}  Top5: {:.3f}  Loss: {:.3f}'.format(
        classerr.value(1), classerr.value(5), loss.mean), flush=True)

    return

#if torch.cuda.is_available():
#    eval_model(test_loader_gpu, quantizer.model, 'cuda')

if torch.cuda.is_available():
    print('Creating CPU copy of Distiller model')
    cpu_model = distiller.make_non_parallel_copy(quantizer.model).cpu()
else:
    cpu_model = quantizer.model
eval_model(test_loader_cpu, cpu_model, 'cpu', print_freq=60)


eval_model(test_loader_cpu, pyt_model, 'cpu', print_freq=60)

if torch.cuda.is_available():
    eval_model(test_loader_gpu, quantizer.model, 'cuda')


