# synextractor
extractor chinese synonyms in large corpus

# Dependency Libs:
tensorflow >= 0.9
sklearn
numpy
scipy

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
