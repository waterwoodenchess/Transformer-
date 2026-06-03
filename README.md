# Transformer-
这个是对Transformer的理论学习

之前在对光场图像处理进行论文学习的时候我说过现在的光场处理还是用的是transformer啊CNN啊这些的。所以今天对transformer进行理论学习，之后可能会有实操学习。两个放在一起吧。

我的学习主要是来源于以下两个：
一个是知乎博主的讲解，我觉得讲的很不错，但是有些地方还是不是很清楚，所以我会在这里面进行深入诠释。这里把链接放上来：https://zhuanlan.zhihu.com/p/338817680
一个是李宏毅老师对于transformer的讲解，我觉得这位老师讲的深度学习都很不错，大家可以多听听~，链接如下：https://www.youtube.com/watch?v=ugWDIIOHtPA&list=PLJV_el3uVTsOK_ZK5L0Iv_EQoL1JefRL4&index=60

原始论文链接：https://arxiv.org/abs/1706.03762

那么我们开始。

一、transformer大致介绍

transformer是在RNN的基础上进行设计的，RNN主要是关注的是全局的信息，就是他必须要看到输入的全部信息才能够输出，而transformertransformer起初是为了进行机器翻译所设计的，他的最开始任务就是把输入的文字序列变成另外一种语言的输出序列，主要做两件事：第一件就是理解输入的语言文字；第二点就是根据理解到的语言意思生成相应的别的语言文字。当然了transformer不只是能做翻译，他还能完成很多的大语言模型工作，比如聊天、写文章、文本分类等等等等，我们现在所用到的GPT啊什么的这些文字工作很多都是基于transformer完成的。除了文字工作，transformer还能够完成图像识别工作，因为到后面大家发现我们也可以把图像看作是一种序列，我们把图像拆分成一个一个的小块，然后按照一定的序列送进transformer里面，transformer就能根据不同小块之间的相关性完成图像识别、图像分类、目标识别等等等等工作。当然我们今天讲解transformer自然是从文字出发进行学习。

为什么transformer会这么厉害？因为transformer里面包含了一个自注意力机制self-attention，这个机制是用来判断一个部分和另外一个部分之间的相关性，判断谁和谁之间更重要。这些我们之后都会讲到。

二、transformer大致结构

在具体学习transformer之前，我觉得我们有必要知道transformer的大致结构。如下图所示：
<img width="640" height="438" alt="image" src="https://github.com/user-attachments/assets/5d26c7a0-9cc1-4726-8e6f-a064d4ff4359" />

这张图表明了一串中文是如何通过不停的encoder和decoder翻译成一串英文的。这些encoder就是transformer的encoder模块，这些decoder就是transformer的decoder模块。原始的transformer包括6个encoder和6个decoder。具体的encoder和decoder如下图所示：
<img width="640" height="884" alt="image" src="https://github.com/user-attachments/assets/375d4972-abcd-45b9-aca8-4bf9cafdb334" />

我们先不要管这张图里面的红框哈，毕竟这张图我也是从别的地方拿过来的。这张图很明白的写明了encoder（左边）和decoder（右边）内部的部分，而最旁边的N×从原始看来就是乘以6，就是我们在第一张大致图里面的6个encoder、6个decoder。所以详细图里每一个蓝色 Encoder，其实都对应大致图左边的一个 Encoder block；每一个粉色 Decoder，其实都对应详细图右边的一个 Decoder block。

当然了，这一张详细图我们表示在进行调试transformer的训练模型，而不是正儿八经进行翻译的模型。

我们可以看到












