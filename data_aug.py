from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode='nearest')

path = 'datasets/data1/train'
numsdict = {}
for item in os.listdir(path):
    itempath = os.path.join(path,item)
    numsdict[item] = len(os.listdir(itempath))
for root,dir,imgs in os.walk(path):
    for name in imgs:
        imgpath = os.path.join(root,name)
        imgsnums = numsdict[imgpath.split('/')[-2]]
        if imgsnums>100:
            numaug = 1 #1
        elif imgsnums> 80:
            numaug = 2 #2
        elif imgsnums > 60:
            numaug = 2 #2

        elif imgsnums > 40:
            numaug = 3 #3
        elif imgsnums > 20:
            numaug = 4 #4

        img = load_img(imgpath)  # 这是一个PIL图像
        x = img_to_array(img)  # 把PIL图像转换成一个numpy数组，形状为(3, 150, 150)
        x = x.reshape((1,) + x.shape)  # 这是一个numpy数组，形状为 (1, 3, 150, 150)

        save_path = 'datasets/data1/other/%s'%(imgpath.split('/')[-2])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # 下面是生产图片的代码
        # 生产的所有图片保存在 `preview/` 目录下
        i = 0
        for batch in datagen.flow(x, batch_size=1,
                                save_to_dir=save_path, save_prefix='ext', save_format='jpg'):
            i += 1
            if i >numaug:
                break  # 否则生成器会退出循环