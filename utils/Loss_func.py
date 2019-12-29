# coding:utf-8
import sys
sys.path.append('./utils')
from torch.nn import CrossEntropyLoss
from cutmix import cutmix
from loss import LabelSmoothSoftmaxCE,FocalLoss
from mixup import mixup_criterion,mixup_data
from torch.autograd import Variable

def cutmix_loss(args,img,label,model,criterion):
    loss = cutmix(args,img,label,model,criterion_base)
    return loss

def mixup_loss(args,img,label,model,criterion):
    inputs, targets_a, targets_b, lam = mixup_data(img, label,args.mixup_alpha)
    inputs, targets_a, targets_b = map(Variable, (inputs,targets_a, targets_b))
    output = model(inputs)
    loss = mixup_criterion(criterion,output,targets_a,targets_b,lam)
    return loss

def lbs_loss(args,img,label,model,criterion):
    criterion = LabelSmoothSoftmaxCE()
    output = model(img)
    loss = criterion(output, label)
    return loss

def focal_loss(args,img,label,model,criterion):
    criterion = FocalLoss(args.num_class)
    output = model(img)
    loss = criterion(output, label)
    return loss

def crossentropy_loss(args,img,label,model,criterion):
    criterion = CrossEntropyLoss()
    output = model(img)
    loss = criterion(output, label)
    return loss

loss_type_dict = {
    'cutmix':cutmix_loss,
    'mixup':mixup_loss,
    'lbs':lbs_loss,
    'focal':focal_loss,
    'crossentropy':crossentropy_loss
}


def loss_func(args,img,label,model):
    criterion = CrossEntropyLoss()
    assert len(args.loss_type) >= 1
    loss = 0
    for loss_type in args.loss_type:
        loss_fn = loss_type_dict[loss_type]
        loss += loss_fn(args,img,label,model,criterion)
    loss /= len(args.loss_type)
    return loss
        
