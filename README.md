# 2020腾讯广告算法大赛

## 一、 解决方案以及不足

### 1.1 数据预处理

### 1.2. 特征工程

### 1.3. 模型训练

### 1.4 方案不足

## 二、代码运行说明

通过`docker build -t 'test:$VERSION' .` 后，启动docker - `docker run -it
--rm --name test:$VERSION /bin/bash` `,
然后在docker环境中以如下pipeline逐步生成最终提交文件.
### 2.1. 数据预处理 

运行 `python3 data_preprocess.py`


### 2.2. 特征工程

运行 `python3 feature_engineering.py`


### 2.3. 模型训练

运行 `python3 train.py` 进行参数选择，模型评估以及生成*.model用于最终预测.

### 2.4. 任务预测

运行 `python3 predict.py` 生成最终针对线上测试集的预测结果.

## 参考学习资料
[1] 2017-2019腾讯广告算法大赛相关代码和数据: <https://blog.csdn.net/fengdu78/article/details/105445921/>

[2] 
