# coding:utf-8
import torch
import torch.nn as nn
from torchvision.models import densenet201,resnet50,resnet18,resnet34,resnet101,densenet121,densenet161,densenet169
from efficientnet_pytorch import EfficientNet

def create_model(args):
    model = None
    if args.model_name == 'densenet201':
        model = Dense201(args)
    elif args.model_name == 'efficientnet_b5':
        model = EfficientNet.from_name('efficientnet-b5')
        model._fc = nn.Sequential(
            # nn.BatchNorm1d(1280),
            nn.Dropout(args.dropout_p),
            nn.Linear(2048,args.num_classes)
        )
    elif args.model_name == 'efficientnet_b4':
        model = EfficientNet.from_pretrained('efficientnet-b4')
        model._fc = nn.Sequential(
            # nn.BatchNorm1d(1280),
            nn.Dropout(args.dropout_p),
            nn.Linear(1792,args.num_classes)
        )
    elif args.model_name == 'efficientnet_b3':
        model = EfficientNet.from_pretrained('efficientnet-b3')
        model._fc = nn.Sequential(
            # nn.BatchNorm1d(1280),
            nn.Dropout(args.dropout_p),
            nn.Linear(1536,args.num_classes)
        )
    elif args.model_name == 'efficientnet_b2':
        model = EfficientNet.from_pretrained('efficientnet-b2')
        model._fc = nn.Sequential(
            # nn.BatchNorm1d(1280),
            nn.Dropout(args.dropout_p),
            nn.Linear(1408,args.num_classes)
        )
    
    elif args.model_name == 'efficientnet_b1':
        model = EfficientNet.from_pretrained('efficientnet-b1')
        model._fc = nn.Sequential(
            # nn.BatchNorm1d(1280),
            nn.Dropout(args.dropout_p),
            nn.Linear(1280,args.num_classes)
        )
    elif args.model_name == 'efficientnet_b0':
        model = EfficientNet.from_pretrained('efficientnet-b0')
        model._fc = nn.Sequential(
            # nn.BatchNorm1d(1280),
            nn.Dropout(args.dropout_p),
            nn.Linear(1280,args.num_classes)
        )
    elif args.model_name == 'resnet101':
        model = resnet101(pretrained=True)
        model.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model.fc = nn.Linear(2048,54)
      
    elif args.model_name == 'resnet50':
        model = resnet50(pretrained=True)
        model.avgpool = nn.AdaptiveAvgPool2d((1,1))
        infeatures = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(args.dropout_p),
            nn.Linear(infeatures,args.num_classes))
    elif args.model_name =='resnet34':
        model = resnet34(pretrained=True)
        infeatures = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(args.dropout_p),
            nn.Linear(infeatures,args.num_classes))
    elif args.model_name =='resnet18':
        model = resnet18(pretrained=True)
        infeatures = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(args.dropout_p),
            nn.Linear(infeatures,args.num_classes))
    elif args.model_name == 'densenet161':
        model = densenet161(pretrained=True)
        infeatures = model.classifier.in_features
        if args.dropout:
            model.classifier=nn.Sequential(
                nn.Dropout(args.dropout_p),
                nn.Linear(infeatures,args.num_classes)
            )
        else:
            model.classifier = nn.Linear(infeatures,args.num_classes)
    elif args.model_name == 'densenet169':
        model =Dense169(args)

    elif args.model_name == 'densenet121':
        model = Dense121()
    elif args.model_name == 'efficientnetb5':
        model = EfficientNet.from_pretrained('efficientnet-b5')
        infeatures = model._fc.in_features
        model.classifier = nn.Sequential(
            nn.BatchNorm1d(infeatures),
            nn.Linear(infeatures,args.num_classes)
        )

    return model


class Dense201(nn.Module):
    def __init__(self,args, n_classes = 54):
        super(Dense201, self).__init__()

        self.net = densenet201(pretrained=True)
        self.net.classifier = nn.Sequential(
            nn.Dropout(p=args.dropout_p),
            nn.Linear(1920,n_classes),
        )
        
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                m.bias.data.zero_()
             

    def forward(self, x):
        return self.net(x)

class Dense169(nn.Module):
    def __init__(self,args, n_classes = 54):
        super(Dense169, self).__init__()

        self.net = densenet169(pretrained=True,)
        self.net.classifier = nn.Sequential(
            nn.Dropout(p=args.dropout_p),
            nn.Linear(1664,n_classes),
        )
        
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                m.bias.data.zero_()             

    def forward(self, x):
        return self.net(x)


class Dense121(nn.Module):
    def __init__(self,n_classes = 54):
        super(Dense121, self).__init__()

        model = densenet121(pretrained=True)
        self.features = model.features
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(1024,n_classes),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                m.bias.data.zero_()             

    def forward(self, x):
        x = self.features(x)
        features = self.avg_pool(x).view(x.size(0),-1)
        print(features.size())
        x = self.classifier(features)

        return x,features


if __name__ == '__main__':
    model = create_model()
