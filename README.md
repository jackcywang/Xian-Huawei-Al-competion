## “华为云杯”2019人工智能创新应用大赛 

链接:[官网](https://competition.huaweicloud.com/information/1000021526/introduction)

比赛介绍：“华为云杯”2019人工智能创新应用大赛，由西安国家民用航天产业基地管理委员会主办、华为公司承办，以“AI在航天，鸿图华构”为主题，是面向全国的人工智能交流赛事。大赛目标是服务以及培养AI人才，构建“用得起、用的好、用得放心”的普惠AI生态，为AI开发者提供一个交流学习、创新挑战的平台。参赛者基于华为云人工智能开发平台ModelArts，根据组委会提供的西安景点、美食、民俗、特产、工艺品等图像数据，进行图像分类模型的开发。

比赛结果 线上测试97.7 排名42/732(top 5%),共1775人参加

数据介绍：数据总共3731张，其中线上测试数据为1000张

![image](https://github.com/jackcywang/Xian-Huawei-Al-competion/raw/master/dataset/data_analyse/train_data.png)
从图中可以看数据很少，且类别不均衡  
数据扩充，从百度爬取了一万多张图片，使用伪标签法（Pseudo label)来扩充图片，使用训练好的模型来预测，阈值选择为97，将预测结果大于97的数据加入到训练数据中

模型选择  
densenet201  
efficientnet_b2  
efficientnet_b3  
efficientnet_b4  
...  
线上得分节点  
densenet201 95.0 densenet201 95.4  
efficientnet_b2 96.6 efficientnet_b2 97.6 efficinetnet_b2 97.7  
efficientnet_b3 95.6 efficientnet_b3 96.0 efficientnet_b3 97.0  
efficientnet_b4 97.5  

训练技巧  
数据增强部分：
1.随机裁剪 randomresizedcrop 
2.随机擦除 random erase
3.mixup
4.水平翻转 
5.归一化  
训练策略：  
学习率使用warmup+CosineAnnealingLR  
采用多尺度训练，分为三段式，图像大小依次递增  
优化器： sgd   
损失函数：  
SmoothCrossEntropyloss+mixup_loss 

涨分点：加大图片分辨率，标签平滑(能抑制过拟合)，mixup(在数据上进行正则)，随机擦除，多尺度训练(增加的模型的泛化能力)，数据清洗（数据是最重要的，一个好的数据集意味着你的起点比别人高很多）  

比赛总结：在前半期的比赛中，由于一直在使用原始数据集进行模型调参，试了大量的模型，验证集精度是始终没法涨到很高，加了许多数据增强的方法，验证集精度提高，线上测试就总是比它点三、四个点，看到自己的排名一天天后退，当时我很崩溃，在知乎上，github,kaggle上到处查阅别人的比赛经验。后来才使用了数据扩充的办法，线上得分才开始慢慢的一点在涨。感谢孙同学，宋同学这一个多月以来的辛苦付出，虽然名次不高，但这次比赛自己也收获了很多。如果要给自己这次比赛打个分的话，我给自己打8分，希望自己在以后的学习，比赛中，能够做到持之以恒的坚持下去，不放弃才是最大的胜利。


