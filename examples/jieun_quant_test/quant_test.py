import distiller
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from distiller.data_loggers import collector_context

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


#####################################
# User-defined model
#####################################
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 3
        for s in size:
            num_features *= s
        return num_features


model_torch = Net()
generate_model_info(model_torch, 'torch_scratch')


#####################################
#torchvision 
#####################################
import torchvision.models as models

model_torchvision = models.resnet18(pretrained=True)
generate_model_info(model_torchvision, 'torchvision_pretrained')
generate_yaml(model_torchvision, 'torchvision_pretrained')

#####################################
# Distiller - parallel=False
#####################################
from distiller.models import create_model
import distiller.quantization as quant
from copy import deepcopy

model_not_parallel = create_model(pretrained=True, dataset='imagenet', arch='resnet18', parallel=False)
generate_model_info(model_not_parallel, 'dist_not_parallel')

#####################################
# Distiller - parallel=True
#####################################
model_parallel = create_model(pretrained=True, dataset='imagenet', arch='resnet18', parallel=True)
generate_model_info(model_parallel, 'dist_parallel')
generate_yaml(model_torchvision, 'dist_parallel')

print("original : ", model_parallel.module.layer1[0].conv1.weight.data[0, 0, :, :])
print('======================')


#####################################
# Quantization
#####################################
model = model_not_parallel

quant_mode = {'activations': 'ASYMMETRIC_UNSIGNED', 'weights': 'SYMMETRIC'}
stats_file = "./stat_yaml/dist_not_parallel.yaml"
#stats_file = "../quantization/post_train_quant/stats/resnet18_quant_stats.yaml"
dummy_input = distiller.get_dummy_input(input_shape=model.input_shape)

quantizer = quant.PostTrainLinearQuantizer(
    deepcopy(model), bits_activations=8, bits_parameters=8, mode=quant_mode,
    model_activation_stats=stats_file, overrides=None
)
temp = quantizer.prepare_model(dummy_input)

pyt_model = quantizer.convert_to_pytorch(dummy_input)
generate_model_info(pyt_model, 'after_quant')

print(pyt_model.layer1[0].conv1.weight().int_repr().data[0, 0, :, :])
print(pyt_model.layer1[0].conv1.weight().dequantize().data[0, 0, :, :])


