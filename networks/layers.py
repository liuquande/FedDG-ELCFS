"""
Wrappers for the operations to take the meta-learning gradient
updates into account.
"""
import torch.autograd as autograd
import torch.nn.functional as F
from torch.autograd import Variable


def linear(inputs, weight, bias, meta_step_size=0.001, meta_loss=None, stop_gradient=False):
    inputs = inputs.cuda()
    weight = weight.cuda()
    bias = bias.cuda()

    if meta_loss is not None:

        if not stop_gradient:
            grad_weight = autograd.grad(meta_loss, weight, create_graph=True)[0]

            if bias is not None:
                grad_bias = autograd.grad(meta_loss, bias, create_graph=True)[0]
                bias_adapt = bias - grad_bias * meta_step_size
            else:
                bias_adapt = bias

        else:
            grad_weight = Variable(autograd.grad(meta_loss, weight, create_graph=True)[0].data, requires_grad=False)

            if bias is not None:
                grad_bias = Variable(autograd.grad(meta_loss, bias, create_graph=True)[0].data, requires_grad=False)
                bias_adapt = bias - grad_bias * meta_step_size
            else:
                bias_adapt = bias

        return F.linear(inputs,
                        weight - grad_weight * meta_step_size,
                        bias_adapt)
    else:
        return F.linear(inputs, weight, bias)

def conv2d(inputs, weight, bias, stride=1, padding=1, dilation=1, groups=1, kernel_size=3):

    inputs = inputs.cuda()
    weight = weight.cuda()
    bias = bias.cuda()

    return F.conv2d(inputs, weight, bias, stride, padding, dilation, groups)


def deconv2d(inputs, weight, bias, stride=2, padding=0, dilation=0, groups=1, kernel_size=None):

    inputs = inputs.cuda()
    weight = weight.cuda()
    bias = bias.cuda()

    return F.conv_transpose2d(inputs, weight, bias, stride, padding, dilation, groups)

def relu(inputs):
    return F.relu(inputs, inplace=True)


def maxpool(inputs, kernel_size, stride=None, padding=0):
    return F.max_pool2d(inputs, kernel_size, stride, padding=padding)


def dropout(inputs):
    return F.dropout(inputs, p=0.5, training=False, inplace=False)

def batchnorm(inputs, running_mean, running_var):
    return F.batch_norm(inputs, running_mean, running_var)


"""
The following are the new methods for 2D-Unet:
Conv2d, batchnorm2d, GroupNorm, InstanceNorm2d, MaxPool2d, UpSample
"""
#as per the 2D Unet:  kernel_size, stride, padding

def instancenorm(input):
    return F.instance_norm(input)

def groupnorm(input):
    return F.group_norm(input)

def dropout2D(inputs):
    return F.dropout2d(inputs, p=0.5, training=False, inplace=False)

def maxpool2D(inputs, kernel_size, stride=None, padding=0):
    return F.max_pool2d(inputs, kernel_size, stride, padding=padding)

def upsample(input):
    return F.upsample(input, scale_factor=2, mode='bilinear', align_corners=False)
