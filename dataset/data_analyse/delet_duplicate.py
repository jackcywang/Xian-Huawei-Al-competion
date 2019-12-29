import hashlib
import os
from PIL import Image
import numpy as np


# files_path = "./all_data"
# save_files_root = "./new_all_data"
# # files = os.listdir(files_path)  # 遍历文件夹下的所有文件

temp = set()  # 创建一个set()
count = 0  # 删除的文件计数
for root, parent,files in os.walk(files_path):
    for file in files:
        save_files_path = os.path.join(save_files_root,root.split('/')[-1])
        if not os.path.exists(save_files_path):
            os.mkdir(save_files_path)
        file_path = os.path.join(root,file)  # 获得完整的路径
        img = Image.open(file_path).convert("RGB")  # 打开图片
        img_array = np.array(img)  # 转为数组
        md5 = hashlib.md5()  # 创建一个hash对象
        md5.update(img_array)  # 获得当前文件的md5码
        if md5.hexdigest() not in temp:  # 如果当前的md5码不在集合中
            temp.add(md5.hexdigest())  # 则添加当前md5码到集合中
            img.save(os.path.join(save_files_path,file))  # 并保存当前图片到保存文件的路径
        else:
            count += 1  # 否则删除图片数加一

print("duplicate removal:", count)  # 最后输出删除图片的总数

# path = 'train'
# des = 'val'
# for root, parent, names in os.walk(des):
#     for name in names:
#         filename = os.path.join(root,name)
#         des_pth = filename.replace(des,path)
#         if os.path.exists(des_pth):
#             print(des_pth)
       
            


