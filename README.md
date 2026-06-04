# Transformer-
这个是对Transformer的理论学习

之前在对光场图像处理进行论文学习的时候我说过现在的光场处理还是用的是transformer啊CNN啊这些的。所以今天对transformer进行理论学习，之后可能会有实操学习。两个放在一起吧。

我的学习主要是来源于以下两个：
一个是知乎博主的讲解，我觉得讲的很不错，但是有些地方还是不是很清楚，所以我会在这里面进行深入诠释。这里把链接放上来：https://zhuanlan.zhihu.com/p/338817680；
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

我们可以看到transformer内部主要是由encoder和decoder构成的。Transformer 的工作流程大体如下：

第一步：获取输入句子的每一个单词的表示向量 X，X=词 Embedding + 位置 Embedding，词 Embedding 表示“这个词本身是什么意思”，比如“苹果”这个词他有水果、公司等等意思；位置 Embedding 表示“这个词出现在第几个位置”，因为 Transformer 本身不像 RNN 那样按顺序一个词一个词读，它同时看整句话，所以必须额外告诉它词的顺序。如果一个句话他有4个词，每个词的长度是6，那么最终给transformer输入的矩阵大小就是4 × 6。

<img width="640" height="263" alt="image" src="https://github.com/user-attachments/assets/6adb0e49-bf18-45f8-b59a-076420752b7f" />

第二步：将得到的单词表示向量矩阵 (如上图所示，每一行是一个单词的表示 x) 传入 Encoder 中，经过 6 个 Encoder block 后可以得到句子所有单词的编码信息矩阵 C，如下图。单词向量矩阵用 X(n×d) 表示， n 是句子中单词个数，d 是表示向量的维度 (论文中 d=512，也就是说一个词用512个长度来进行表示)。每一个 Encoder block 输出的矩阵维度与输入完全一致。

<img width="640" height="900" alt="image" src="https://github.com/user-attachments/assets/0eb09c9a-9fac-4ffb-8ac9-12e608185fbb" />

第三步：我们将已经在Encoder内部得到的信息矩阵C传递给decoder，然后decoder会根据这个C来生成英文。decoder在进行翻译的时候不是直接进行翻译的，而是一个一个进行逐步翻译的：
输入 <Begin>          -> 预测 I
输入 <Begin> I        -> 预测 have
输入 <Begin> I have   -> 预测 a
输入 <Begin> I have a -> 预测 cat

<img width="640" height="370" alt="image" src="https://github.com/user-attachments/assets/aaf54e8f-d4a5-43d9-a62f-e10abd98ddb6" />

在进行这一步操作的时候，我们通常会使用mask对第i个后面的词汇进行遮挡。就是我们对第i个词汇进行预测的时候，我们只能依据之前已经拿到的或者预测好了的词汇进行预测，我们不知道后面的词汇是什么，也没有办法依赖第i个后面的词汇进行预测。详细的我们之后会讲。

三、Transformer的输入

之前我们就知道，transformer的输入是由词的embedding和位置的embedding相加构成的。

<img width="640" height="142" alt="image" src="https://github.com/user-attachments/assets/93466155-d1bd-464e-a06c-eaf9743e7abd" />

1.词汇的embedding

词汇的embedding是什么之前已经说过了，就是代表了这个词汇的意思是什么。而单词的 Embedding 有很多种方式可以获取，例如可以采用 Word2Vec、Glove 等算法预训练得到，也可以在 Transformer 中训练得到。

2.位置的embedding

对于transformer来说，必须要知道一句话每个词汇的位置，因为Transformer 一次性看整句话，不像 RNN 那样从左到右读，所以它天然不知道顺序。比如：我 打 他；他 打 我。这两个句子的词差不多，但顺序不同，意思完全不一样。所以 Transformer 需要额外加入“位置信息”。

在公式中，我们用PE表示位置embedding：

<img width="640" height="136" alt="image" src="https://github.com/user-attachments/assets/4ec641e1-1df4-43f8-9740-664efcb29713" />

其中，pos 表示单词在句子中的位置，d 表示 PE的维度 (与词 Embedding 一样)，2i 表示偶数的维度（用sin），2i+1 表示奇数维度 (用cos)。什么意思？比如一句话：我有一只猫。总共四个词：我、有、一只、猫，“我”的pos是0、“有”的pos是1、“一只”的pos是2、“猫”的pos是3。假设每个词用6个长度或者维度进行表示，那么每一个词汇的PE(pos) = [第0维, 第1维, 第2维, 第3维, 第4维, 第5维]，而每一个维度都是按照公式规定来计算：第 0 维：PE(pos, 0) = sin(...)、第 1 维：PE(pos, 1) = cos(...)；第 2 维：PE(pos, 2) = sin(...)、第 3 维：PE(pos, 3) = cos(...)；第 4 维：PE(pos, 4) = sin(...)、第 5 维：PE(pos, 5) = cos(...)

使用这种公式进行位置的计算有好处：1.可以表示任意长度的位置：如果训练时最长句子是 20 个词，但测试时来了 21 个词，公式仍然可以算出第 21 个位置的位置编码；2.方便模型理解相对位置：比如模型不只需要知道“这个词在第 5 位”，还需要知道“这个词离另一个词差 3 个位置”，sin/cos 有一个性质：pos + k 的编码可以由 pos 的编码推出来，因为 Sin(A+B) = Sin(A)Cos(B) + Cos(A)Sin(B), Cos(A+B) = Cos(A)Cos(B) - Sin(A)Sin(B) 。所以模型更容易学到“相隔 k 个词”的关系。

四、Encoder结构

<img width="640" height="884" alt="image" src="https://github.com/user-attachments/assets/4427a047-3a83-48aa-b8ff-31fa85dc3057" />

如上图所示，是transformer的具体架构图，其中红框圈出来的部分就是transformer的encoder结构。它包含了一个multi-head attention、两个add&norm和一个feed forward。我们之后会把这一部分拆开来仔细讲一遍。

1. multi-head attention

multi-head attention是由多层自注意力机制self attention组成的，而自注意力机制是transformer里面最重要的部分。所以我们很有必要先把self attention搞清楚。

1.1 自注意力机制 self attention

<img width="705" height="884" alt="image" src="https://github.com/user-attachments/assets/0180f410-869c-462f-ac4b-8f54366289f6" />

上面这张图表示的是self attention的结构图，我们可以看到在自注意力机制里面它的输入主要依靠的是Q、K和V，分别叫做查询、键值和值，在self attention里面它主要接收到的是来自输入X或者encoder模块带来的值，而Q、K和V正是通过self attention的线性变换输入得到的。具体的来讲这三个字母都是什么意思：Q：我现在想找什么信息？ K：我这里有什么信息可以被别人匹配？ V：如果别人关注我，我真正提供什么内容？

1.1.1 Q、K、V的计算

首先，我们在之前已经得到了这个句子的表示矩阵X，由于self attention的输入基本需要用X进行表示，所以我们在对self attention进行输入的时候就可以使用线性变阵矩阵WQ,WK,WV计算得到Q,K,V，计算如下图所示。

<img width="640" height="875" alt="image" src="https://github.com/user-attachments/assets/662687b6-071c-497c-9cb9-5a3b5d344057" />

有人会问WQ、WK、WV是怎么得到的？在transformer里面，这三个矩阵是可训练优化的权重矩阵。在最开始这三个矩阵完全是随机值，但是会在训练中将这三个矩阵的数值进行优化，大概的流程是这样的：1. 随机生成 WQ、WK、WV、2. 用它们算出 Q、K、V、3. 模型根据 Q、K、V 做 attention，然后预测结果、4. 预测错了，就根据 loss 反向传播、5. 更新 WQ、WK、WV、6. 重复很多次

1.1.2 self-attention的输出

在得到QKV之后我们就能计算self-attention的输出了。计算公式如下：

<img width="640" height="171" alt="image" src="https://github.com/user-attachments/assets/f17465a5-69a2-4505-a519-5848a853c449" />

首先，先对K进行转置，然后和Q相乘得到QKᵀ，表示了每个词和每个词之间的相关程度。为什么QKᵀ能够表示这个意思？因为两个向量相乘，本质上是在做点积，如果两个向量方向很像，点积通常比较大；如果方向不相似，点积会比较小，甚至可能是负数。所以点积又能代表两个向量之间的关系是近还是远。所以QKᵀ的意思就是每两个词之间的Q和K的相关性，说的更明白一点，可以把 QKᵀ 理解成一句话：每个词拿着自己的“问题 Q”，去和所有词的“标签 K”做匹配，看谁最相关。比如：

QKᵀ 得到一个 4 × 4 矩阵：
 
      K1   K2   K3   K4
Q1   [ 8    2    1    7 ]

Q2   [ 3    9    2    5 ]

Q3   [ 1    2    8    6 ]

Q4   [ 4    5    7    9 ]

第一行就表示第一个词的Q和其他所有词的K的相关程度。

<img width="640" height="228" alt="image" src="https://github.com/user-attachments/assets/c62c0c4f-bebb-4883-b5cc-0432c4b74867" />

其次，因为通常计算的数据都比较大，为了防止内积过大，我们在这里面除以√dk。其中dk是QKV的维度。

之后，我们对QKᵀ/√dk进行softmax操作。softmax可以理解成把一组数字变成一组“比例/概率”，让QKᵀ/√dk矩阵的每一行相加都等于1，原来分数越大，softmax 后占比越大，因为原来的分数越高，代表着二者的匹配度越高，在进行softmax转化为注意力比例的时候相应的也就需要变大。


<img width="640" height="247" alt="image" src="https://github.com/user-attachments/assets/d63f6fb4-20d2-4cf1-9e19-af85f38c510d" />

最后，因为我们现在已经得到了每个词对所有词的注意力占比，所以为了得到这一个词的输出，我们应该用占比×每个单词的内容向量V，这就是公式里面在进行softmax操作完毕之后乘以V的原因。

<img width="640" height="217" alt="image" src="https://github.com/user-attachments/assets/e87497ad-1b6a-4f15-b17c-062789358e21" />

举个例子：比如我们现在知道了一个矩阵的第一行是[0.3, 0.2, 0.2, 0.3]，这就意味着单词 1 关注 单词 1 的比例是 0.3、单词 1 关注 单词 2 的比例是 0.2、单词 1 关注 单词 3 的比例是 0.2、单词 1 关注 单词 4 的比例是 0.3，注意这几个数加起来等于 1：0.3 + 0.2 + 0.2 + 0.3 = 1。然后怎么算单词 1 的输出 Z1？就是用第 1 行的 attention 权重，对所有词的 V 向量做加权求和：Z1 = 0.3 × V1 + 0.2 × V2 + 0.2 × V3 + 0.3 × V4

意思是：单词 1 最终的新表示= 保留 30% 单词 1 自己的信息+ 吸收 20% 单词 2 的信息+ 吸收 20% 单词 3 的信息+ 吸收 30% 单词 4 的信息。用矩阵表示就是下图所示。

<img width="640" height="216" alt="image" src="https://github.com/user-attachments/assets/86971699-d8cd-414b-9ae5-bf4955221ff7" />

那这个输出Z就代表了融合了其他词信息的这个词，意思就是，原来的这个词只是包含了这个词可能有的意思（词embedding）和位置信息（位置embedding），经过self-attention之后这个词融合了其他词的相关信息，变得更加准确了。比如两句话：A:苹果 很 甜；B：苹果 发布 手机，对于苹果这个词来说，原来的向量差不多，但是经过self-attention之后，因为融合了其他的词汇含义，所以对于A句，苹果 更多关注的是 甜，所以这个 苹果 更趋向于水果；而对于B句，苹果 更多关注的是 手机，所以这个苹果更多的是指公司名称。变得更加理解上下文语境含义了。

1.2 multi-head attention

之前我们就说过了multi-head attention是由多层自注意力机制self attention组成的，具体的结构我们看下面的图：

<img width="640" height="859" alt="image" src="https://github.com/user-attachments/assets/c82b2d47-f7fa-4475-8d08-73a0bd5bcc09" />

我们在进行multi-head attention的时候，会将X传递给 h 个不同的 Self-Attention 中。这h个不同的Self-Attention都有自己的WQ、WK、WV，会学习到不同的关注方式。对于一个Self-Attention来说，只能学习到一种方式来看句子，但是一个句子可以从不同的方式来看，所以multi-head attention的好处，也是目的就在于可以让模型从不同的方式去理解学习。

<img width="640" height="831" alt="image" src="https://github.com/user-attachments/assets/8c2f7205-9ffe-421f-a52c-9e9c04073286" />

在X经过这么多个不同的 Self-Attention 得到了h个不同的输出Z之后，我们会用concat把他们拼接在一起，得到[head1的信息 | head2的信息 | ... | head8的信息]，然后经过linear操作对这些信息进行整合学习，比如学习：哪些 head 的信息更重要、哪些维度应该组合起来、最终输出应该长什么样等等。矩阵操作就是用concat好的Z矩阵和linear矩阵进行相乘，得到最终的输出Z。注意，linear矩阵也是可训练参数，和WQ、WK、WV一样，一开始随机初始化，然后训练中通过反向传播学习出来。

<img width="640" height="388" alt="image" src="https://github.com/user-attachments/assets/f7aac58f-63ee-4f25-8674-a013c4f65055" />

所以，Multi-Head Attention 就是多个 Self-Attention 并行工作，让模型从多个角度看句子；Concat 把多个结果拼起来；最后的 Linear 层负责把这些不同角度的信息融合成最终输出。

2.1 ADD & Norm

ADD & Norm分了两个部分：一个是ADD，一个是Norm。首先来看ADD，ADD操作其实是一种残差连接，具体的表示如下图所示。在transformer里面，ADD就是把multi-head attention的输入和输出加在一起。然后进行后面的layernorm操作。残差连接的意义在于没有丢弃掉原来的信息x，而是在原来的基础上加一点变化。比如 Attention 处理后得到 F(X)，它可能提取了上下文关系。
但原来的词本身信息 X 也很重要，所以直接加回来。

<img width="640" height="117" alt="image" src="https://github.com/user-attachments/assets/c3a5377a-7d6e-444b-abca-38b47bd8be99" />

由于ADD & Norm在encoder里面用到了两次，所以他的计算公式就是：A = LayerNorm(X + MultiHeadAttention(X))；B = LayerNorm(A + FeedForward(A))。

那什么又是layernorm？layernorm的作用就是把某一个向量的数值稍微稳定一下，让他不要偏离的太过离谱，比如一个向量的数值太大、分布不稳定，所以LayerNorm 会把它变成均值接近 0、方差接近 1 的形式，作用是：让训练更稳定、让模型更容易收敛、避免一层层传下去数值越来越乱。

2.2 Feedforward

FeedForward，通常是两层全连接层（全连接层的意思就是输出的每一个数和输入的每一个数都有关。比如：输入：[x1, x2, x3, x4]，输出：[y1, y2, y3]，其中：

y1 = x1*w11 + x2*w21 + x3*w31 + x4*w41 + b1，y2 = x1*w12 + x2*w22 + x3*w32 + x4*w42 + b2，y3 = x1*w13 + x2*w23 + x3*w33 + x4*w43 + b3，y1/y2/y3 每个输出都用到了所有输入，所以叫全连接）。FeedForward的公式是：FFN(X) = max(0, XW1 + b1)W2 + b2，其中：XW1 + b1：第一层线性变换；max(0, XW1 + b1)：ReLU 激活函数；再乘 W2 + b2：第二层线性变换。

FeedForward的目的在于让被attention融合了的词汇进行深入加工学习，学习更深层的特征以及更深层的判断标准和规则，以提升模型的表达能力。attention进行的是词汇和词汇之间的向量变换，而feedforward进行的是每个词汇内部的向量变换。

以上的这些东西就组成了encoder模块，接受输入矩阵X后，经过多个encoder模块输出叠加形成输出矩阵C，这个矩阵C之后会用在decoder里面

<img width="640" height="900" alt="image" src="https://github.com/user-attachments/assets/150c5dce-dca8-4585-81e4-e96fc5af2f9d" />

五、decoder结构
decoder详细结构图如下图所示：

<img width="640" height="884" alt="image" src="https://github.com/user-attachments/assets/6f260b48-56d9-446e-bcde-884fad93f7cf" />

上图红色部分为 Transformer 的 Decoder block 结构，与 Encoder block 相似，但是存在一些区别：

包含两个 Multi-Head Attention 层；第一个 Multi-Head Attention 层采用了 Masked 操作；第二个 Multi-Head Attention 层的K, V矩阵使用 Encoder 的编码信息矩阵C进行计算，而Q使用上一个 Decoder block 的输出计算；最后有一个 Softmax 层计算下一个翻译单词的概率。

5.1 第一个 Multi-Head Attention

在decoder的这个Multi-Head Attention中，我们用到了masked和teacher forcing的概念。
首先我们来讲一下masked。一般来讲在翻译的过程中是顺序翻译的，即翻译完第 i 个单词，才可以翻译第 i+1 个单词。通过 Masked 操作可以防止第 i 个单词知道 i+1 个单词之后的信息。

比如：
输入 <Begin>          -> 预测 I
输入 <Begin> I        -> 预测 have
输入 <Begin> I have   -> 预测 a
输入 <Begin> I have a -> 预测 cat

这之前都已经说过了。但是这样的顺序翻译还是太慢了，所以我们在decoder训练的时候可以用到teacher forcing进行并行化训练。

接下来我们讲一下用teacher forcing进行并行化训练，也就是把正确的语句传递给decoder。

在进行并行化训练的时候我们是同时根据mask之后的语句进行下一步预测。比如，我现在把<Begin>I have a cat给decoder，在进行并行化训练的时候，我们这样做：

1.看到<Begin>  可能预测I You He...... 正确语句（之前已经给了decoder）是I（随后修正阶段根据loss进行修正）

2.看到<Begin> I（这是在正确语句上面给后面语句加的mask，不管之前预测的是什么，在这里我们只给正确的语句） 可能预测am like......正确语句是have（同上）

3.看到<Begin> I have（同上）可能预测a an some......正确语句是a（同上）

4. 5.同上。注意12345这几步是同时进行的，之后在修正阶段用loss对目前猜测的答案进行修正。这样就是用teacher forcing进行并行性训练，再说明白一点就是给正确的前文然后进行并行性训练。当然了我们在这里只是用这一句话举例子表明整个过程，在实际训练的时候我们会有很多语句对模型进行训练，这样模型就会学习语言的表达形式、语句的上下文含义等等。在实际进行翻译或者进行大语言模型工作的时候就会根据之前预测出来的语句进行下一步预测。

我们再从矩阵的角度看一下整个流程：

第一步，这里仍然使用I have a cat为例。输入矩阵包含 "<Begin> I have a cat" (0, 1, 2, 3, 4) 五个单词的表示向量，Mask 是一个 5×5 的下三角矩阵。mask中我们看到第0个单词只能看第0个单词，第1个单词可以看第1个单词和第0个单词的意思，不能看到后面的所有单词。

<img width="640" height="277" alt="image" src="https://github.com/user-attachments/assets/329b6626-e5db-4995-bfb6-80393eeb7645" />

第二步：接下来的操作和之前的 Self-Attention 一样，通过输入矩阵X计算得到Q,K,V矩阵。然后计算Q和Kᵀ 的乘积QKᵀ。

<img width="640" height="240" alt="image" src="https://github.com/user-attachments/assets/cc8bacbc-7ec1-45dc-86ae-cf32a4b96008" />

第三步：在得到QKᵀ之后需要进行 Softmax，计算 attention score，我们在 Softmax 之前需要使用Mask矩阵遮挡住每一个单词之后的信息，遮挡操作如下：

<img width="640" height="206" alt="image" src="https://github.com/user-attachments/assets/44647b26-6749-4230-a8b8-4719a55a842b" />

然后对maskQKᵀ进行softmax操作。

有人会好奇为什么一个mask矩阵就能进行遮挡了呢？在softmax之前，这个mask矩阵是个下三角矩阵，他的上三角部分数值极小。而在softmax之后，mask的上三角部分为0，也就是说第0个单词对其他第1234个单词的注意力为0，第1个单词对其余的第234个单词的注意力也为0，只关注自己和前面的单词。等于看不到。

第四步：使用 Mask QKᵀ与矩阵 V相乘，得到输出 Z，则单词 1 的输出向量 Z1 是只包含单词 1 信息的，其余同理。

<img width="640" height="292" alt="image" src="https://github.com/user-attachments/assets/4f0e0cab-dd43-405a-bf2e-ec31718fb20b" />

第五步：通过上述步骤就可以得到一个 Mask Self-Attention 的输出矩阵 Zi，然后和 Encoder 类似，通过 Multi-Head Attention 拼接多个输出Zi，然后计算得到第一个 Multi-Head Attention 的输出Z，Z与输入X维度一样。

5.2 第二个Multi-Head Attention

这个Multi-Head Attention和第一个没啥区别，只是第二个的Q 来自 Decoder、K、V 来自 Encoder 输出 C。意思就是：第二个Multi-Head Attention在进行语言翻译的时候，不仅能看到前文已经翻译好的单词，还能看到原来语言的全文。比如预测 cat 时：Decoder 当前状态：我现在要根据 “I have a” 继续翻译、Encoder 信息：我 / 有 / 一只 / 猫，这时候 Cross-Attention 可能会重点关注中文里的：猫，于是更容易预测出：cat。

这两个Multi-Head Attention的区别在于，第一个只能看到翻译好的前文，第二个不仅能看到前文，还能看到原来的全文。再说明白点：第一个 Masked Self-Attention 负责让 Decoder 知道：我前面已经说了什么？现在英文句子进行到什么语法状态？哪些前文词会影响当前位置？ 第二个 attention 的作用是：根据当前英文状态，去中文原文里找相关信息

目标英文输入：<Begin> I like eating
      
        ↓

第一个 attention：Masked Self-Attention，整理英文前文内部关系

        ↓

第二个 attention：Cross-Attention，拿整理后的英文状态去看中文 Encoder 输出 C

        ↓

预测下一个词

5.3 Softmax 预测输出单词

Decoder block 最后的部分是利用 Softmax 预测下一个单词，在之前的网络层我们可以得到一个最终的输出 Z，因为 Mask 的存在，使得每个单词的输出 Z只包含之前单词和本单词的信息，如下：

<img width="640" height="239" alt="image" src="https://github.com/user-attachments/assets/f119a36d-0471-4aa4-8ef4-e1e49b559dbf" />

在得到这个信息矩阵之后，我们会用linear对每个单词进行“打分操作”（这是因为linear内部包含有和训练的权重矩阵W以及参数b，会用大量训练把这些参数调整好，最终学会给合适的词最高分），然后softmax会把分数变成概率，然后选最高概率的词作为预测词汇。


完结~撒花~~~






