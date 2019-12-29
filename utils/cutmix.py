import numpy as np
import torch

def cutmix(args,img,label,model,criterion):
    r = np.random.rand(1)
    if args.beta > 0 and r < args.cutmix_prob:
        lam = np.random.beta(args.beta, args.beta)
        rand_index = torch.randperm(img.size()[0]).cuda()
        target_a = label
        target_b = label[rand_index]
        bbx1, bby1, bbx2, bby2 = rand_bbox(img.size(), lam)
        img[:, :, bbx1:bbx2, bby1:bby2] = img[rand_index, :, bbx1:bbx2, bby1:bby2]
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img.size()[-1] * img.size()[-2]))
        # compute output
        input_var = torch.autograd.Variable(img, requires_grad=True)
        target_a_var = torch.autograd.Variable(target_a)
        target_b_var = torch.autograd.Variable(target_b)
        output1 = model(input_var)
        loss = criterion(output1, target_a_var) * lam + criterion(output1, target_b_var) * (1. - lam)
    else:
        input_var = torch.autograd.Variable(img, requires_grad=True)
        target_var = torch.autograd.Variable(label)
        output1 = model(input_var)
        loss = criterion(output1, target_var)  
    return losss

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2