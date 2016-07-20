# Synextractor
extractor chinese synonyms in large corpus

# Dependency Libs:
tensorflow >= 0.9
sklearn
numpy
scipy

############
# 模型效果 #
############
这里包括的两个模型对于同义词的挖掘效果测试如下，其中测试集通过随机预先随机抽取
syn_pos.txt文件中约18000个同义词对组成正样本，随机采样词对约180000对组成负样本。

dim=50, window=2, min_count=100:


| Precison | Recall | F1 Score |
|----------|--------|----------|
| 0.7751   | 0.4975 | 0.6060   |


dim=100, window=5, min_count=100:


| Precison | Recall | F1 Score |
|----------|--------|----------|
| 0.8157   | 0.4771 | 0.6010   |

一些成功挖掘出的同义词


| 原同义词词林　　　　　　  |　补充　　  |
|---------------------------|------------|
|　群众，民众，大众　　　　 |　百姓　　  |
|　侨民，侨胞　　　　　　　|　华侨　　   |
|　工薪层，工薪阶层　　　　|　工薪族　|
|　笑哈哈　笑呵呵　笑眯眯　|　乐呵呵　|

不足的部分：
对于一词多义的同义词挖掘

# Usage:
############
# 模型训练 #
############
1.生成数据

python train.py -d [词向量维度] -w [上下文窗口] --min_count [最小词频] --generate_data -f [语料文件] -p [正样本] -t [线程数] 

2.模型训练

python train.py -d [词向量维度] -w [上下文窗口] --min_count [最小词频] --train_model -o [返回模型配置文件路径] --npoch [模型迭代次数] 

3.数据 + 训练

python train.py -d [词向量维度] -w [上下文窗口] --min_count [最小词频] --generate_data -f [语料文件]  -p [正样本] -t [线程数] --train_model -o [返回模型配置文件路径] --npoch [模型迭代次数]

4.清除指定的模型和数据

python train.py -d [词向量维度] -w [上下文窗口] --min_count [最小词频] --clean

############
# 模型使用 #
############

注：目前只支持加载两个模型

1.直接返回同义词

python synonym.py --m1 [模型I] --m2 [模型II] -n [返回同义词的最大个数，推荐10及以上] --synonym -w [单词]

2.给定词性标注的一列词，生成同义词词典

python synonym.py --m1[模型I] --m2 [模型II] -n [返回同义词的最大个数，推荐10及以上] --thesaurus -f [词性标注过的词列] -o [输出的路径]
