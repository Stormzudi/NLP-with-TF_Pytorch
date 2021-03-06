# 1 句法分析

- 句法结构分析（成分结构分析，短语结构分析）
  - 完全句法分析（完全短语结构分析）
  - 局部分析（浅层分析）
- 依存关系分析（依存句法分析，依存结构分析，依存分析）


## 1.1 句法结构分析概述

### 1.1.1 基本概念

- 句法分析：对输入的单词序列（一般为句子）判断其构成是否符合给定的语法，分析出合乎语法的句子的句法结构。
- 句法分析树（分析树）：句法结构一般用树状数据结构表示。
- 句法结构分析器（分析器）：完成这种分析过程的程序模块。
- 任务：
  - 判断输入的字符串是否属于某种语言
  - 消除输入句子中词法和结构方面的歧义
  - 分析输入句子的内部结构，如成分结构，上下文关系等

### 1.1.2 语法形式化

- 上下文无关文法（CFG）
- 基于约束的文法（合一语法）：
  - 功能合一语法
  - 树链接语法
  - 词汇功能语法
  - 广义的短语结构语法
  - 中心语驱动的短语结构语法

### 1.1.3 基本方法

- 基于规则的分析方法
  - 基本思路：由人工组织语法规则，建立语法知识库，通过条件约束和检查来实现句法结构歧义的消除
  - 分类：
    - 自顶向下的分析方法
    - 自底向上的分析方法
    - 两者相结合的分析方法

- 基于统计的分析方法
  - 语法驱动的分析方法的基本思路：由生成语法定义被分析的语言及其分析出的类别，在训练数据中观察到的各种语言现象的分布以统计数据的方式与语法规则一起编码。当遇到歧义情况时，统计数据用于对多种分析结果的排序或选择。

# 2 语义分析

语义分析的基本任务：

- 词：词义消歧
- 句子：语义角色标注
- 篇章：指代消歧（共指消解），篇章语义分析

## 2.1 语义消歧概述

- 基本观点：一个词的不同语义一般发生在不同的上下文中。在有监督的消歧方法中，可以根据训练数据得知一个多义词所处的不同上下文与特定词义的对应关系，那么，多义词的词义识别问题实际上就是该词的上下文分类问题，一旦确定了上下文所属的类别，也就确定了该词的词义类型。因此，有监督的学习通常也称为分类任务。在无监督的词义消歧中，由于训练数据未经标注，因此，首先需要利用聚类算法对同一个多义词的所有上下文进行等价类划分，如果一个词的上下文出现在多个等价类中，那么，该词被认为是多义词。然后，在词义识别时，将该词的上下文与其各个词义对应上下文的等价类进行比较，通过上下文对应等价类的确定来断定词的词义。因此，无监督的学习通常称为聚类任务。
- 伪词：为了避免手工标注的困难，人们通常采用制造人工数据的方法来获得大规模训练数据和测试数据，这些制造出来的人工数据称为伪词。（将两个自然词汇合并）

# 3 文本分类与情感分类

## 3.1 文本分类概述

- 获得这样一个函数$\Phi:D\times C \rightarrow\{T,F\}$，其中$D=\{d_1,d_2,…，d_{|D|}\}$表示需要进行分类的文档，$C=\{c_1,c_2,…，c_{|C|}\}$表示预定义的分类体系下的类别集合。T值表示对于$<d_j,c_i>$来说，文档$d_j$属于$c_i$，而F值表示对于$<d_j,c_i>$来说，文档$d_j$不属于$c_i$。
- 最终目的：找到一个有效的映射函数，准确的实现域$D\times C$到T或F的映射，这个映射函数实际上就是我们通常说的分类器。
![- <img src="D:\picture\image-20210209102908519.png" alt="image-20210209102908519" style="zoom:80%;" />](https://img-blog.csdnimg.cn/20210210122349274.png#pic_center)

- 两个关键问题：
  - 文本的表示
  - 分类器设计
- 分类：
  - 基于知识工程的分类系统
  - 基于机器学习的分类系统

## 3.2 文本表示

- 向量空间模型VSM的基本概念：

  - 文档：通常是文章中具有一定规模的片段，如句子、句群，段落、段落组直至整篇文章。
  - 项 / 特征项：特征项是 VSM 中最小的不可分的语言单元，可以是字、词、词组或短语等。一个文档的内容被看成是它含有的特征项所组成的点、集合，表示为∶$Document=D（t_1，t_2，…，t_n）$，其中$t_k$是特征项，$1\leq k\leq n$。
  - 项的权重：对于含有n个特征项的文档 $D（t_1，t_2，…，t_n）$，每一特征项$t_K$都依据一定的原则被赋予一个权重 $w_k$，表示它们在文档中的重要程度。这样一个文档D可用它含有的特征项及其特征项所对应的权重所表示∶ ，简记$D=D（t_1，w_1;t_2，w_2;…，t_n,w_n）$为 $D=D（w_1,w_2,…w_n）$，其中 $w_k$就是特征项$t_k$的权重，$1\leq k\leq n$。

- 向量空间模型：

  给定一个文档$D（t_1，w_1;t_2，w_2;…，t_n,w_n）$，D符合以下两条规定：

  - 各个特征项$t_k(1\leq k\leq n)$互异（即没有重复）
  - 各个特征项$t_k$无先后顺序关系（即不考虑文档的内部结构）

  在以上两条约定下，可以把特征项$t_1，t_2，…，t_n$看成一个n维坐标系，而权重$w_1,w_2,…w_n$为相应的坐标值。因此一个文本就表示为n维空间的一个向量。我们称$D=D（w_1,w_2,…w_n）$为文本D的向量表示或向量空间模型。

  ![<img src="D:\picture\image-20210209105108352.png" alt="image-20210209105108352" style="zoom: 50%;" />](https://img-blog.csdnimg.cn/20210210122409951.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)


- 向量的相似性度量：

  任意两个文档$D_1,D_2$之间的相似系数$Sim(D_1,D_2)$指两个文档内容的相关程度。设文档
  $$
  D_1=D_1（w_1,w_2,…w_n）\\
  D_2=D_2（w_1,w_2,…w_n）
  $$
  则：
  $$
  Sim(D_1,D_2)=\sum^n_{k=1}w_{1k}\times w_{2k}
  $$
  考虑归一化：
  $$
  Sim(D_1,D_2)=cos\theta=\frac{\sum^n_{k=1}w_{1k}\times w_{2k}}{\sqrt{\sum^n_{k=1}w_{1k}^2\sum^n_{k=1}w_{2k}^2}}
  $$
  
- 基本步骤：

  - 根据训练样本集生成文本表示所需的特征序列$D=\{t_1,t_2,…，t_d\}$
  - 依据文本特征项序列，对训练文本集和测试样本集的各个文档进行权重赋值、规范化等处理，将其转化为机器学习算法所需的特征向量。