# 2020腾讯广告算法大赛
广告受众基础属性预估 - 根据用户在不同广告上的反馈，同时预测出其年龄和性别  

* [参赛手册](./docs/2020腾讯广告算法大赛参赛手册.pdf)
* [原始输入数据说明](./docs/2020腾讯广告算法大赛数据说明.xlsx)

## 一、 解决方案以及不足

### 1.1 数据预处理

### 1.2. 特征工程

### 1.3. 模型训练

### 1.4 方案不足

## 二、环境准备
docker version - 18.09.7

nvidia-docker version - 2.2.2

git

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

### 2.5. 数据集以及运行组件下载
从百度网盘下载数据集以及spark运行组件压缩文件, 数据集解压后放于项目的一级目录data文件夹下(Ps.data文件夹需新建)  

data网盘地址: https://pan.baidu.com/s/1oXMXVRs5i7lgkF-Yr9aTag 提取码: jey3  

spark运行组件网盘地址: https://pan.baidu.com/s/1cKp-jIPtcVlTyDmOmV03zA 提取码: i9f4

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

### 2.7. 在container中配置spark运行环境
`
cp /home/spark_env/.bashrc ~/.bashrc
`

## 三、代码运行说明

以如下pipeline逐步生成最终提交文件

### 3.1. 数据格式化以及预处理 

`cd mlpipeline/preprocess`

`python3 data_reformat.py`

`python3 data_preprocess.py`


### 3.2. 特征工程

`cd mlpipeline/feature_engineering`

`python3 feature_engineering.py`


### 3.3. 参数选择, 模型评估以及训练

`python3 train.py`

### 3.4. 任务预测以及生成提交结果于submission文件夹(Ps.需新建)

`python3 predict.py` 

## 参考学习资料
[1] 2019腾讯广告算法大赛round1冠军github: <https://github.com/guoday/Tencent2019_Preliminary_Rank1st>

[2] 2019腾讯广告算法大赛round2冠军github: <https://github.com/bettenW/Tencent2019_Finals_Rank1st>

[3] 易观性别年龄预测第一名解决方案github: <https://github.com/chizhu/yiguan_sex_age_predict_1st_solution>

[4] 华为用户人口属性（i.e. 年龄段）预测解决方案github:
<https://github.com/luoda888/HUAWEI-DIGIX-AgeGroup>
