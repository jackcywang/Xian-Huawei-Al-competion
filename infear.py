# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from tensorboardX import SummaryWriter
from utils import *
import numpy as np
import time
import datetime
import argparse
import os
import os.path as osp
from Dataset import Xian_Dataset,Pseudo_Dataset
import os
import shutil
from Creat_model import create_model
os.environ['CUDA_ENABLE_DEVICES'] = '0'
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from efficientnet_pytorch import EfficientNet

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--test_batch_size',type=int,default=1)
    parse.add_argument('--num_workers', type=int, default=8)
    parse.add_argument('--model_name',type=str,default='efficientnet_b3')
    parse.add_argument('--dropout_p',type=float,default=0.6)
    parse.add_argument('--dataset',type=str,default='./datasets/data1/val')
    parse.add_argument('--model_out_name',type=str,default='model_out/efficientnet_b3/10_lr_0.0005__wd_0.0001_drop_0.5_mix_0_1225/model_best.pth')
    return parse.parse_args()

label_id_names={
    "0": "工艺品/仿唐三彩",
    "1": "工艺品/仿宋木叶盏",
    "2": "工艺品/布贴绣",
    "3": "工艺品/景泰蓝",
    "4": "工艺品/木马勺脸谱",
    "5": "工艺品/柳编",
    "6": "工艺品/葡萄花鸟纹银香囊",
    "7": "工艺品/西安剪纸",
    "8": "工艺品/陕历博唐妞系列",
    "9": "景点/关中书院",
    "10": "景点/兵马俑",
    "11": "景点/南五台",
    "12": "景点/大兴善寺",
    "13": "景点/大观楼",
    "14": "景点/大雁塔",
    "15": "景点/小雁塔",
    "16": "景点/未央宫城墙遗址",
    "17": "景点/水陆庵壁塑",
    "18": "景点/汉长安城遗址",
    "19": "景点/西安城墙",
    "20": "景点/钟楼",
    "21": "景点/长安华严寺",
    "22": "景点/阿房宫遗址",
    "23": "民俗/唢呐",
    "24": "民俗/皮影",
    "25": "特产/临潼火晶柿子",
    "26": "特产/山茱萸",
    "27": "特产/玉器",
    "28": "特产/阎良甜瓜",
    "29": "特产/陕北红小豆",
    "30": "特产/高陵冬枣",
    "31": "美食/八宝玫瑰镜糕",
    "32": "美食/凉皮",
    "33": "美食/凉鱼",
    "34": "美食/德懋恭水晶饼",
    "35": "美食/搅团",
    "36": "美食/银枸杞炖耳",
    "37": "美食/柿子饼",
    "38": "美食/浆水面",
    "39": "美食/灌汤包",
    "40": "美食/烧肘子",
    "41": "美食/石子饼",
    "42": "美食/神仙粉",
    "43": "美食/粉汤羊血",
    "44": "美食/羊肉泡馍",
    "45": "美食/肉夹馍",
    "46": "美食/荞面饸饹",
    "47": "美食/菠菜面",
    "48": "美食/蜂蜜凉粽子",
    "49": "美食/蜜饯张口酥饺",
    "50": "美食/西安油茶",
    "51": "美食/贵妃鸡翅",
    "52": "美食/醪糟",
    "53": "美食/金线油塔"
}

despath = '/dataset/ext_data/'

Lables = []
for key in label_id_names.keys():
    Lables.append(key)

#根据训练模型获得伪标签数据
def infer_Pseudo(args):
    datapath,label_list,_ = get_data(args,mode='val')
    val_set = Pseudo_Dataset(args,datapath,label_list)
    val_loader = DataLoader(val_set,
                            batch_size=args.test_batch_size,
                            drop_last=False,
                            shuffle=True,
                            pin_memory=True,
                            num_workers=args.num_workers)

    
    net = create_model(args)
    net.load_state_dict(torch.load(args.model_out_name))
    net.cuda()

    net.eval()

    
    with torch.no_grad():
        for img,names in val_loader:
            img = img.cuda()
            size = img.size(0)
            outputs = net(img)
            outputs = F.softmax(outputs, dim=1)
            max_prob = np.max(outputs.cpu().numpy())
            index = np.argmax(outputs.cpu().numpy())
            
            if max_prob >= 0.95:
                print(max_prob,index)
                predicted = torch.max(outputs, dim=1)[1].item()
                print(predicted,names)
                
                for i in range(size):
                    pre_name = str(predicted)
                    true_name = names[i].split('/')[-2]
                    des_dir = os.path.join(despath,true_name)
                    if not os.path.exists(des_dir):
                        os.mkdir(des_dir)
                    if not osp.exists(osp.join(des_dir,names[i].split('/')[-1])):
                        if pre_name == true_name:
                            shutil.copy(names[i],osp.join(des_dir,names[i].split('/')[-1]))

                        msg = '{} {}'.format(true_name, pre_name)
                        print(msg)
                    


    print('----------Done!----------')

#评估模型验证集结果
def evaluate_val(args):
    datapath,label_list,_ = get_data(args,mode='val')
    val_set = Xian_Dataset(args,datapath,label_list,mode='val')
    val_loader = DataLoader(val_set,
                            batch_size=args.test_batch_size,
                            drop_last=False,
                            shuffle=True,
                            pin_memory=True,
                            num_workers=args.num_workers)

    
    net = create_model(args)
    net.load_state_dict(torch.load(args.model_out_name))
    net.cuda()

    net.eval()

    total = 0
    correct = 0
    net.eval()
    pre = []
    true = []
    with torch.no_grad():
        for img, lb in val_loader:
            img, lb = img.cuda(), lb.cuda()
            outputs = net(img)
            outputs = F.softmax(outputs, dim=1)
            predicted = torch.max(outputs, dim=1)[1]
            #predicted = torch.argsort(outputs[0], descending=True)[:1][0]
            total += lb.size()[0]
            # print(predicted)
            correct += (predicted == lb).sum().cpu().item()
            pre.append(predicted.item())
            true.append(lb.item())
    
    print('correct:{}/{}={:.4f}'.format(correct, total, correct * 1. / total))

    print('----------Done!----------')
    conf_matrix = confusion_matrix(true,pre)
    plot_confusion_matrix(conf_matrix,labels=Lables)

#混淆矩阵
def plot_confusion_matrix(cm, labels):
    cmap=plt.cm.binary
    tick_marks = np.array(range(len(labels))) + 0.5
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(20, 15), dpi=120)
    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)
    intFlag = 0 # 标记在图片中对文字是整数型还是浮点型
        
    for x_test, y_test in zip(x.flatten(), y.flatten()):
        #

        if (intFlag):
            c = cm[y_test][x_test]
            plt.text(x_test, y_test, "%d" % (c,), color='red', fontsize=8, va='center', ha='center')

        else:
            c = cm_normalized[y_test][x_test]
            if (c > 0.01):
                #这里是绘制数字，可以对数字大小和颜色进行修改
                plt.text(x_test, y_test, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')
            else:
                plt.text(x_test, y_test, "%d" % (0,), color='red', fontsize=7, va='center', ha='center')
    if(intFlag):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
    else:
        plt.imshow(cm_normalized, interpolation='nearest', cmap=cmap)
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.title('')
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('Index of True Classes')
    plt.xlabel('Index of Predict Classes')
    plt.savefig('confusion_matrix.jpg', dpi=300)

if __name__ == '__main__':
    args = parse_args()
    args.data_val = './datasets/data/val'
    args.num_classes=54
    args.imgsize = 224
    # infer_Pseudo(args)
    evaluate_val(args)
