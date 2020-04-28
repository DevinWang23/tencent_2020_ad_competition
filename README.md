# 2020腾讯广告算法大赛

## 一、 解决方案以及不足

### 1.1 数据预处理

### 1.2. 特征工程

### 1.3. 模型训练

### 1.4 方案不足

## 二、环境准备
**docker version - 18.09.7**
**nvidia-docker version - 2.2.2**

### 2.1. 登录镜像管理帐号

```
docker login \
--username=wmq王王 \  
registry.cn-shenzhen.aliyuncs.com
``` 
(密码:wang3436)

### 2.2. 拉取docker

```
docker pull registry.cn-shenzhen.aliyuncs.com/mengqiu/machine_learning:pytorch1.4-cuda10.1-py3-spark2.4.5
```

### 2.3. cd到项目目录下, 通过Dockerfile构建docker

```
docker build -t 'tencent_2020_ad_competition:latest' .
``` 

### 2.4. clone代码

```
git clone https://github.com/DevinWang23/tencent_2020_ad_competition.git
```

### 2.5. 下载数据集以及spark运行环境, 将数据集放于项目的data文件夹下
``

### 2.6. 运行docker环境
```
nvidia-docker run \
-it \
--rm \
--name ad_competition \
-p 4112:4112 \
-p 18080:18080 \ 
-p 4040:4040 \
-v /etc/localtime:/etc/localtime \
-v $DOWNLOAD_SPARK_ENV_PATH:/home/spark_env \
-v $PROJECT_PATH:/home/Tecent_2020_Ad_Competition \
registry.cn-shenzhen.aliyuncs.com/mengqiu/machine_learning:pytorch1.4-cuda10.1-py3-spark2.4.5 \
/bin/bash
```

## 三、代码运行说明

以如下pipeline逐步生成最终提交文件

### 3.1. 数据预处理 

运行 `python3 data_preprocess.py`


### 3.2. 特征工程

运行 `python3 feature_engineering.py`


### 3.3. 模型训练

运行 `python3 train.py` 进行参数选择，模型评估以及生成*.model用于最终预测.

### 3.4. 任务预测

运行 `python3 predict.py` 生成最终针对线上测试集的预测结果.

## 参考学习资料
[1] 2019腾讯广告算法大赛初赛冠军github: <https://github.com/guoday/Tencent2019_Preliminary_Rank1st>
[2] 2019腾讯广告算法大赛复赛冠军github: <https://github.com/bettenW/Tencent2019_Finals_Rank1st>
