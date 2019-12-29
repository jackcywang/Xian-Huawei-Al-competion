# coding:utf-8
import os
import json
import argparse
from tqdm import tqdm
import numpy as np
import warnings
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch import optim
from tensorboardX import SummaryWriter
#dataset
from Dataset import get_data,Xian_Dataset,my_collate_fn,Multi_scale
from sklearn.model_selection import StratifiedKFold,StratifiedShuffleSplit
#utils
from utils.Loss_func import loss_func
from utils.LR_schedule import LinearScheduler
from utils.early_stop import EarlyStopping
#models
from Creat_model import create_model
import numpy as np


parse = argparse.ArgumentParser(description='Xian Huawei_cloude_competetion')

# img
parse.add_argument('--dataset_path', type=str, default='/media/wang/D78DEE6C30580B32/project/Xian_Al/data')
parse.add_argument('--num_classes', type=int, default=54)
parse.add_argument('--imgsize',type=int,default=224)
parse.add_argument('--multi_scale', type=bool, default=True)
parse.add_argument('--multi_size',type=list,default=[280,380,480])
parse.add_argument('--multi_milestone',type=list,default=[40,80])

# train
parse.add_argument('--pre_epoch', type=int, default=2)      
parse.add_argument('--warmup_epoch', type=int, default=1)   
parse.add_argument('--normal_epoch', type=int, default=60)  
parse.add_argument('--batch_size', type=int, default=64)
parse.add_argument('--num_workers', type=int, default=4)
parse.add_argument('--pretrained', type=bool, default=True)
parse.add_argument('--cuda', type=bool, default=True)
parse.add_argument('--gpu_id', type=str, default='0')
parse.add_argument('--dropout', type=bool, default=False)
parse.add_argument('--dropout_p', type=float, default=0.5)
parse.add_argument('--model_name', type=str, default='efficientnet_b2')

# optimizer
parse.add_argument('--optimizer', type=str, default='sgd')
parse.add_argument('--lr', type=float, default=1e-3)
parse.add_argument('--min_lr', type=float, default=1e-6)
parse.add_argument('--lr_fc_times', type=int, default=1, help='the lr of fc layer times')
parse.add_argument('--lr_scheduler', type=str, default=None)
parse.add_argument('--momentum', type=float, default=0.9, metavar='M')
parse.add_argument('--weight_decay', type=float, default=1e-4) #5e-4
parse.add_argument('--nesterov', type=bool, default=True)

# criterion
parse.add_argument('--loss_type', type=str, default=['crossentropy'])

#data regularization method
parse.add_argument('--mixup',type=bool,default=False,help='mixup regularization')
parse.add_argument('--mixup_alpha',type=float,default=0.5,help='mixup regularization')
parse.add_argument('--lbs',type=bool,default=False,help='labelsmoothing regularization')
parse.add_argument('--cutmix',type=bool,default=False,help='cutmix regularization')
parse.add_argument('--beta',type=float,default=1.0)
parse.add_argument('--cutmix_prob',type=float,default=0.5)

# save pth .pth
parse.add_argument('--save_path', type=str, default=None)

args = parse.parse_args()
warnings.filterwarnings("ignore")


def train_model(args):
    # try:
        # device
    global best_acc
    global schedule

    writer = SummaryWriter(logdir=args.sub_tensorboard)
    set_device_environ(args)

    # model
    model = create_model(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    #criterion
    criterion = nn.CrossEntropyLoss()
    
    # model resume training
    if args.resume:
        model.load_state_dict(torch.load('model_out/efficientnet_b4/10_lr_0.0005__wd_0.0001_drop_0.5_mix_0_1225/model_best.pth'))
    
    device = torch.device('cuda:0' if torch.cuda.is_available() and args.cuda==True  else 'cpu')
    model.to(device)

    parallel = False
    if args.cuda:
        if len(args.gpu_id.split(',')) > 1 and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            parallel = True

    optimizer = optim.SGD(,lr=args.lr,momentum=0.9, nesterov=args.nesterov, weight_decay=args.weight_decay)
    #Split_data and Train
    train_imgs_path,train_labels,samples_each_class = get_data(args,'train')
    val_imgs_path,val_labels,_ = get_data(args,'val')
    
    Train_dataset = Xian_Dataset(args,train_imgs_path,train_labels,mode='train')
    Val_dataset = Xian_Dataset(args,val_imgs_path,val_labels,mode='val')
    
    print('train:',len(Train_dataset),'val:',len(Val_dataset))
    train_loader = DataLoader(Train_dataset,batch_size = args.batch_size,
                                pin_memory = True, num_workers = args.num_workers,shuffle=True,drop_last=True
                                ,collate_fn=my_collate_fn)
    val_loader = DataLoader(Val_dataset,batch_size = args.batch_size,
                                pin_memory = True, num_workers = args.num_workers,shuffle=False,drop_last=False
                                ,collate_fn=my_collate_fn)
    # lr_scheduler
    if args.pre_epoch:
        print('Starting pre_training...')  
        for name, param in model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False 
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                momentum=0.9,nesterov=args.nesterov,weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.normal_epoch*len(train_loader), eta_min=args.min_lr)
    # training
    best_acc = 0
    
    for epoch in range(0,args.normal_epoch+args.warmup_epoch):

        model.train()
        loss_sum = 0.0

        if args.multi_scale:
            Train_dataset = Multi_scale(args,train_imgs_path,train_labels,epoch,mode='train')
            Val_dataset = Multi_scale(args,val_imgs_path,val_labels,epoch,mode='val')
            train_loader = DataLoader(Train_dataset,batch_size = args.batch_size,
                                pin_memory = True, num_workers = args.num_workers,shuffle=True,drop_last=True
                                ,collate_fn=my_collate_fn)
            val_loader = DataLoader(Val_dataset,batch_size = args.batch_size,
                                pin_memory = True, num_workers = args.num_workers,shuffle=False,drop_last=False
                                ,collate_fn=my_collate_fn)
        
        if epoch == args.pre_epoch: 
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=0.9,nesterov=args.nesterov,weight_decay=args.weight_decay)
            if args.warmup_epoch:
                scheduler = LinearScheduler(optimizer, start_lr=args.min_lr, end_lr=args.lr, all_steps=args.warmup_epoch*len(train_loader))
                print('-------- start warmup for {} epochs -------'.format(args.warmup_epoch))

        if epoch == args.pre_epoch + args.warmup_epoch:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.normal_epoch*len(train_loader), eta_min=args.min_lr)
            print('-----------start normal train for {} epoch ------'.format(args.normal_epoch))


        tbar = tqdm(train_loader, ncols=200)
        true_label = 0
        total = 0
        for sampel in train_loader:
            img,lable = sampel
        
        for i, sample in enumerate(tbar):
            # get a batch_size sample
            img, label = sample

            img = img.to(device)
            label = label.to(device)

            # optimizer to clean the gradient buffer
            optimizer.zero_grad()
            # forward
            output = model(img)

            loss = loss_func(args,img,label,model)
            loss.backward()

            # optimize
            optimizer.step()

            
            scheduler.step()

            loss_sum += loss.item()
            lr = optimizer.param_groups[0]['lr']

            _, predicted = torch.max(output.data, 1)  # 预测类别
            true_label += (predicted == label).sum().item()
            total += img.size(0)
            acc = true_label/total

            tbar.set_description('epoch:%d, train loss:%.4f,train acc:%.4f,lr:%f ' % (epoch, loss_sum/(i+1),acc,lr))

            # loss_sum = 0
        train_loss, train_accuracy= 0.0, 0.0
        test_loss, test_accuracy= 0.0, 0.0
        

        # train and val the training model
        kwargs = {'model':model, 'criterion':criterion, 'device':device}
        test_loss, test_accuracy = train_val_model(args, val_loader, **kwargs)
        train_loss,train_accuracy = train_val_model(args,train_loader,**kwargs)

        #scheduler.step(test_loss)
        
        # early_stoping(test_loss)
        # if early_stoping.early_stop:
        #     print("early stoping !!!")
        #     break

        print(('test_loss: %.4f, test_accuracy: %.4f\n') %(test_loss, test_accuracy))

        writer.add_scalars('acc',{'train_acc':train_accuracy,'test_acc':test_accuracy},epoch)
        writer.add_scalars('loss',{'train_loss':train_loss,'test_loss':test_loss},epoch)
        

        if test_accuracy >best_acc:
            best_acc = test_accuracy
            if parallel:
                torch.save(model.module.state_dict,args.save_path+'/model_best.pth')
            else:
                torch.save(model.state_dict(), args.save_path+'/model_best.pth')
    writer.close()
        

def train_val_model(args, dataloader=None, model=None, criterion=None, device=None):
    model.eval()
    total = 0
    loss = 0.0  
    true_label = 0  
    accuracy = 0.0  
    with torch.no_grad():
        for sample in dataloader:
            img, label = sample
            img = img.to(device)
            label = label.to(device)

            output = model(img)

            batch_size_loss = criterion(output, label)
            loss += batch_size_loss

            total += label.size()[0]

            _, predicted = torch.max(output.data, 1)  
            true_label += (predicted == label).sum().item()

        loss /= len(dataloader)
        accuracy = true_label / total

        return loss.item(), accuracy 


# device
def set_device_environ(args):
    # gpu
    if os.name == 'nt':
        print('system: windows')
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        args.gpu_id = '0'

    elif os.name == 'posix':
        print('system: linux')
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        args.gpu_id = '0'


if __name__ == '__main__':
    # dataset
    args.data_train = 'datasets/data2/train'
    args.data_val = 'datasets/data2/val'
    args.cuda = True
    args.pretrained = True
    args.val = False
    args.pre_epoch = 0
    args.warmup_epoch=5
    args.normal_epoch = 150
    args.lr = 1e-3
    args.min_lr = 1e-6
    args.batch_size = 16
    args.mixup_alpha = 0.2
    args.resume = False
    args.weight_decay = 1e-4
    args.loss_type = ['crossentropy','mixup'] # ['crossentropy', 'focal', 'lbs','cutmix','mixup']
    args.optimizer = 'sgd' 
    args.model_name ='efficientnet_b2'  # ['resnet101', 'se_resnet', 'resnet50', 'densenet121', 'densenet169', 'densenet201' 'efficientnet_b7']

    args.dropout = True  
    args.dropout_p = 0.5 
    args.tensorboard = './tensorboard'

    args.sub_tensorboard = os.path.join(args.tensorboard,args.model_name,str(args.batch_size)+'_lr_'+str(args.lr)+'_'\
                    +'_wd_'+str(args.weight_decay)+'_drop_'+str(args.dropout_p)+'_mixup_'+str(args.mixup_alpha)+'_1225')

    args.save_path = './model_out/'+args.model_name+'/'+str(args.batch_size)+'_lr_'+str(args.lr)+'_'\
                    +'_wd_'+str(args.weight_decay)+'_drop_'+str(args.dropout_p)+'_mixup_'+str(args.mixup_alpha)+'_1225'
    print('args.pre_epoch',args.pre_epoch)
    print('args.warmup_epoch',args.warmup_epoch)
    print('args.normal_epoch',args.normal_epoch)
    print('optimizer:', args.optimizer)
    print('dropout:', args.dropout, ' p:', args.dropout_p)
    print('model_name',args.model_name)
    print('batch_size:', args.batch_size)
    print('weight_decay:',args.weight_decay)
    print('lr:',args.lr)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if not os.path.exists(args.sub_tensorboard):
        os.makedirs(args.sub_tensorboard)
    train_model(args)


