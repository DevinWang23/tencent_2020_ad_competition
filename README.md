# 2020腾讯广告算法大赛
广告受众基础属性预估 - 根据用户历史点击广告行为，同时预测出其年龄和性别，以性别准确率 + 年龄准确率为评估指标，最终线上1.47659， ranking 37/1008.  

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

### 3.4. 任务预测以及生成提交结果于submission文件夹(Ps.submission文件夹需新建)

`python3 predict.py` 

## 参考学习资料
[1] 2019腾讯广告算法大赛round1冠军github: <https://github.com/guoday/Tencent2019_Preliminary_Rank1st>

[2] 2019腾讯广告算法大赛round2冠军github: <https://github.com/bettenW/Tencent2019_Finals_Rank1st>

[3] 易观性别年龄预测第一名解决方案github: <https://github.com/chizhu/yiguan_sex_age_predict_1st_solution>

[4] 华为用户人口属性（i.e. 年龄段）预测解决方案github:
<https://github.com/luoda888/HUAWEI-DIGIX-AgeGroup>

[5] 2020腾讯广告算法大赛冠军github: <https://github.com/guoday/Tencent2020_Rank1st>

[6] 2020腾讯广告算法大赛冠军知乎: <https://zhuanlan.zhihu.com/p/166710532>

[7] 2020腾讯广告算法大赛亚军知乎：https://zhuanlan.zhihu.com/p/185045764
