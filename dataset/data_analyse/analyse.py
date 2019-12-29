import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import os
des = 'train'
dataset={}
for dirs in os.listdir(des):
    dirpath = osp.join(des,dirs)
    nums = len(os.listdir(dirpath))
    dataset[dirs]=nums
   
# 构建数据
x_data=sorted([int(key) for key in dataset.keys()])
y_data =[dataset[str(i)] for i in x_data]
num = np.sum(np.array(y_data))


print(x_data,y_data)

plt.bar(x=x_data, height=y_data, label='total nums %s'%(str(num)),color='steelblue', alpha=0.8)

for x, y in enumerate(y_data):
    plt.text(x, y, '%s' % y, ha='center', va='bottom')


plt.title('The distrubution of dataset')

plt.xlabel('labels')
plt.ylabel("numbers")

plt.legend()
plt.show()
