import os
import os.path as osp
import shutil
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
oripath =  r'E:\project\Xian_Al\model_v6\datasets\data1\preview'
des =r'E:\project\Xian_Al\model_v6\datasets\data1\val'
widths = []
heights = []
all = [oripath,des]
for path in all:
    for root,dirs,files in os.walk(path):
        for image in files:
            imgpath = osp.join(root,image)
            img = Image.open(imgpath)
            width, height = img.size[0],img.size[1]
            if width<850 or height<850:
                widths.append(width)
                heights.append(height)


x = np.array(widths)
y = np.array(heights)

width=int(np.mean(x))
height = int(np.mean(y))

print('avrage width: %s'%(width))
print('avrage height: %s'%(height))

plt.figure()

#设置标题
plt.title('Scatter Plot')
#设置X轴标签
plt.xlabel('width')
#设置Y轴标签
plt.ylabel('height')
#画散点图
plt.scatter(x,y,c = 'r',marker = '+')
#设置图标
plt.text(2500,2500,'width {}'.format(str(width)))
plt.text(2500,2400,'height {}'.format(str(height)))
# plt.legend('height %s'%(str(height)))
#显示所画的图
plt.savefig('width_and_height')


        




