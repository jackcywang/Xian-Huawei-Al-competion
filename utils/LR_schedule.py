# coding:utf-8
class LinearScheduler:
    def __init__(self,optimizer, start_lr, end_lr, all_steps):
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.all_steps = all_steps
        self.cur_step = 0
    def step(self):
        self.cur_step += 1
        if self.cur_step>=self.all_steps:
            self.cur_step=self.all_steps
        cur_lr = (self.end_lr-self.start_lr) * (self.cur_step*1./self.all_steps) + self.start_lr
        for param in self.optimizer.param_groups:
            param['lr'] = cur_lr

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch<= 28:
        lr = args.lr
    elif epoch <=48:
        lr = 1e-4
    elif epoch <= 70:
        lr = 1e-5
    elif epoch <= 100:
        lr = 1e-6
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr