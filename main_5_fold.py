# coding:utf-8
import os
import json
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch import optim
from tensorboardX import SummaryWriter
# dataset
from Dataset import get_data,Xian_Dataset
from sklearn.model_selection import StratifiedKFold,StratifiedShuffleSplit

from Loss_func import loss_func
from utils.LR_schedule import LinearScheduler
from utils.early_stop import EarlyStopping


from Creat_model import create_model
import numpy as np

parse = argparse.ArgumentParser(description='Xian Huawei_cloude_competetion')

# img
parse.add_argument('--dataset_path', type=str, default='/media/wang/D78DEE6C30580B32/project/Xian_Al/all_data')
parse.add_argument('--num_classes', type=int, default=54)
parse.add_argument('--test_size', type=int, default=600)
parse.add_argument('--input_size', type=int, default=224)
parse.add_argument('--n_splits', type=int, default=5)

# train
parse.add_argument('--normal_epoch', type=int, default=80)  
parse.add_argument('--batch_size', type=int, default=32)
parse.add_argument('--num_workers', type=int, default=4)
parse.add_argument('--pretrained', type=bool, default=True)
parse.add_argument('--cuda', type=bool, default=True)
parse.add_argument('--gpu_id', type=str, default='0')
parse.add_argument('--dropout', type=bool, default=False)
parse.add_argument('--dropout_p', type=float, default=0.5)
parse.add_argument('--model_name', type=str, default='densenet201')

# optimizer
parse.add_argument('--optimizer', type=str, default='sgd')
parse.add_argument('--lr', type=float, default=1e-3)
parse.add_argument('--min_lr', type=float, default=1e-6)
parse.add_argument('--lr_scheduler', type=str, default=None)
parse.add_argument('--momentum', type=float, default=0.9, metavar='M')
parse.add_argument('--weight_decay', type=float, default=1e-4) #5e-4
parse.add_argument('--nesterov', type=bool, default=True)

# criterion
parse.add_argument('--loss_type', type=str, default=['crossentropy'])

# save pth .pth
parse.add_argument('--save_path', type=str, default=None)
args = parse.parse_args()



def train_model(args):
    # device
    global best_acc
    global schedule

    writer = SummaryWriter(logdir=args.sub_tensorboard)
    set_device_environ(args)

     # model
    

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    criterion = nn.CrossEntropyLoss().cuda()

    print(torch.cuda.is_available())
    # device = torch.device('cuda:'+args.gpu_id if torch.cuda.is_available() and args.cuda==True  else 'cpu')
    device = torch.device('cuda:0' if torch.cuda.is_available() and args.cuda==True  else 'cpu')

    image_paths,labels = get_data(args,'train')

    skf = StratifiedShuffleSplit(n_splits = args.n_splits,test_size=args.test_size,random_state = 2019)

    best_accuracy = {}

    #K折交叉验证
    for fold,(train_idx, val_idx) in enumerate(skf.split(image_paths, labels)):
        train_imgs_path, train_labels = list(np.array(image_paths)[train_idx]),list(np.array(labels)[train_idx])
        val_imgs, val_labels = list(np.array(image_paths)[val_idx]), list(np.array(labels)[val_idx])

        Train_dataset = Xian_Dataset(args,train_imgs_path,train_labels,mode='train')
        Val_dataset = Xian_Dataset(args,val_imgs,val_labels,mode='val')
        print('train:',len(Train_dataset),'val:',len(Val_dataset))
        train_loader = DataLoader(Train_dataset,batch_size = args.batch_size,
                                    pin_memory = True, num_workers = args.num_workers,shuffle=True)
        val_loader = DataLoader(Val_dataset,batch_size = args.batch_size,
                                    pin_memory = True, num_workers = args.num_workers,shuffle=False)
        
        early_stoping = EarlyStopping()

        model = create_model(args)
        model.to(device)


    
        # training
        best_acc = 0
        
        for epoch in range(0,args.normal_epoch):

            model.train()
            loss_sum = 0.0

            tbar = tqdm(train_loader, ncols=200)  # 进度条
            for i, sample in enumerate(tbar):
                # get a batch_size sample
                img, label = sample

                img = img.to(device)
                label = label.to(device)

                # optimizer to clean the gradient buffer
                optimizer.zero_grad()
                output = model(img)

                loss = loss_func(args,img,label,model)
                # backward
                loss.backward()

                # optimize
                optimizer.step()

                

                # 每个batch_size 打印 train_loss
                loss_sum += loss.item()
                lr = optimizer.param_groups[0]['lr']

                tbar.set_description('epoch:%d, train loss:%.4f,lr:%f ' % (epoch, loss_sum/(i+1),lr))

                # loss_sum = 0
            train_loss, train_accuracy= 0.0, 0.0
            test_loss, test_accuracy= 0.0, 0.0
            

        
            # train and val the training model
            kwargs = {'model':model, 'criterion':criterion, 'device':device}
            train_loss, train_accuracy = train_val_model(args, train_loader, **kwargs)
            test_loss, test_accuracy = train_val_model(args, val_loader, **kwargs)

            scheduler.step(test_loss)

            # early_stoping(test_loss)
            # if early_stoping.early_stop:
            #     print("early stoping !!!")
            #     break
   
            print(('train_loss: %.4f, train_accuracy: %.4f, test_loss: %.4f, test_accuracy: %.4f\n') %
                (train_loss, train_accuracy, test_loss, test_accuracy))

            writer.add_scalars('acc',{'train_acc':train_accuracy,'test_acc':test_accuracy},epoch)
            writer.add_scalars('loss',{'train_loss':train_loss,'test_loss':test_loss},epoch)
            
            
            

            if test_accuracy > best_acc:
                
                best_acc = test_accuracy
                torch.save(model.state_dict(), args.save_path+'/model_best_fold_{}.pth'.format(str(fold+1)))
        
        
        best_accuracy['fold_{}'.format(str(fold+1))]=best_acc
        
        #model.load_state_dict(torch.load(args.save_path+'/model_best_fold_{}.pth'.format(str(fold+1))))
    writer.close()
    with open('result.json', 'w') as f:
        json.dump(best_accuracy, f, indent=2)
    print('**********************************')   
    for key in best_accuracy.keys():
        print(key,best_accuracy[key])
    print('**********************************')  
        

# 训练过程中的model对 train_dataloader 和 val_dataloader 进行验证
def train_val_model(args, dataloader=None, model=None, criterion=None, device=None):
    model.eval()
    total = 0
    loss = 0.0  # 损失值
    true_label = 0  # 准确数
    accuracy = 0.0  # 准确率
    # each_class_true_label = [0]*54 # 每个类别的准确率

    with torch.no_grad():
        # 训练集测试
        for sample in dataloader:
            img, label = sample
            img = img.to(device)
            label = label.to(device)

            # 预测分数
            output = model(img)

            # 计算损失值
            batch_size_loss = criterion(output, label)
            loss += batch_size_loss

            total += label.size()[0]

            # 计算准确率
            _, predicted = torch.max(output.data, 1)  # 预测类别
            true_label += (predicted == label).sum().item()

        loss /= len(dataloader)
        accuracy = true_label / total

        return loss.item(), accuracy 


# device
def set_device_environ(args):
    # gpu
    if os.name == 'nt':
        print('system: windows')
        os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
        args.gpu_id = '0, 1, 2, 3'

    elif os.name == 'posix':
        print('system: linux')
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        args.gpu_id = '0'


if __name__ == '__main__':
    # dataset
    args.dataset = './dataset/train'
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
    args.loss_type = ['lbs','mixup'] # ['crossentropy', 'focal', 'lbs','cutmix','mixup']
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