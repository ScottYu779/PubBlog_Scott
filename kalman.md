卡尔曼滤波器(kf)(kalman filter)完整学习笔记(包括理解和推导)
---

# 数学和控制基础
## 数学符号
  $\hat{a}$:上面的尖号，表示最大似然估计,读“a caret”
  $\tilde{X}$:上面带有浪号，真实值$X$与估计值$\hat{X}$之差，读音“a tilde”
  $\overline a$:表示平均值，读音"a bar"
  $a^{'}$,读音为a prime
## 矩阵的迹
  矩阵的迹：矩阵主对角线上的所有元素之和称之为矩阵的迹
## 详解误差几个概念
### 方差
  - 方差是用来衡量一组数据离散程度的统计量，是变量离期望值的距离的统计。误差的平方的平均值。
  - 是误差的二次方的平均值
  - $ \sigma ^2=\frac{1}{n}\sum_i^n(x_i-\overline x)^2 $
    $\overline x$平均数
### 均方差=标准差
  - 标准差（standard deviation)等同于均方差（mean square error, MSE）
  - 方差的平均值，方差是平方的平均值，
  - 误差的一次方的均值
  $\sigma=\sqrt{\frac{1}{n}\sum_i^n(x_i-\overline x)^2}$
  
### 协方差
  - 可以通俗的理解为：两个变量在变化过程中是同方向变化？还是反方向变化？同向或反向程度如何？  
    你变大，同时我也变大，说明两个变量是同向变化的，这时协方差就是正的。  
    你变大，同时我变小，说明两个变量是反向变化的，这时协方差就是负的。  
    从数值来看，协方差的数值越大，两个变量同向程度也就越大。反之亦然。
  - 和方差的关系
    协方差矩阵的主对角线元素分别为两个变量的方差
  - 公式
    $Cov(X,Y) = E[(X - u_x)(Y - u_y)]$

## 控制论基础
  - ****状态方程****
    $x_k = A_k x_{(k-1)} + w_{(k-1)}$
  - **测量方程**
    $y_k = C_k x_k + v_k$
  - **符号解释**
    $w$过程噪声
    $v$测量噪声

# 公式讲解
## 卡尔曼滤波的理解要义
  - 卡尔曼滤波算法核心思想在于预测+测量反馈，它由两部分组成，第一部分是 线性系统状态预测方程，第二部分是 线性系统观测方程。
  - 预测就是基于模型的预测，类似于控制里边的前馈，或者先验知识，反馈、观测和后验则是实际情况下的结果，将这两者进行迭代性质的处理，根据性能实时分配权重系数(滤波的本质就是权重的分配)，所以也称之为滤波。所以既可以叫滤波算法，也可以叫控制算法。
  - 卡尔曼的滤波的目的是综合测量值和预测值得到一个最优值
  - 卡尔曼是一个迭代过程，上一次的值根据权重分配来影响下一次的值，这也是一种惯性滤波了
## 公式
### 版本一
  - **状态方程**
    $X_{(k|k-1)} = A*X_{(k-1|k-1)} + B*u_{(k)} + W_{(k)}$
  - **观测方程**
    $Z_{(k|k-1)} = H*X_{(k|k-1)} + V_{(k)}$
  - **根据卡尔曼增益参数来对估计值和测量值进行权重分配的公式**
    $X_{(k|k)} = X_{(k|k-1)} + K_{(k)}*(Z_{(k)} - Z_{(k-1)}) = X_{(k|k-1)} + K_{(k)}*(Z_{(k)}-H*X_{(k|k-1)})$
  - **协方差矩阵**
    $P_{(k|k)} = E[(X_{(k)} - X_{(k|k)}) (X_{(k)} - X_{(k|k)})^T]$
### 版本二
  - **状态方程**
    线性系统状态预测方程的表达式，假设过程激励噪声满足高斯分布
    $x_{k} = A*x_{k-1} + B*u_{(k-1)} + w_{(k-1)}$
    - $p(w) = N(0,Q)$
    - $A$:表示状态转移系数矩阵，$m * n$阶
    - $B$:表示可选的控制输入的增益矩阵
    - $Q$:表示过程激励噪声的协方差矩阵
  - **线性系统观测方程**
    线性系统观测方程表达式，假设测量噪声矩阵满足高斯分布
    $z_{k} = H*x_{k} + v_{k}$
    - $p(v) = N(0,R)$ p表示概率
    - $H$:测量系数矩阵，$m * n$阶矩阵
    - $R$:表示测量噪声协方差矩阵
  - **根据卡尔曼增益参数来对估计值和测量值进行权重分配的公式**
    $X_{(k|k)} = X_{(k|k-1)} + K_{(k)}*(Z_{(k)} - Z_{(k-1)}) = X_{(k|k-1)} + K_{(k)}*(Z_{(k)}-H*X_{(k|k-1)})$
  - **协方差矩阵**
    $P_{(k|k)} = E[(X_{(k)} - X_{(k|k)}) (X_{(k)} - X_{(k|k)})^T]$
### 版本三
  - **状态方程**
    $X_{(k|k-1)} = A*X_{(k-1|k-1)} + B*u_{(k)} + W_{(k)}$
  - **观测方程**
    $Z_{(k|k-1)} = H*X_{(k|k-1)} + V_{(k)}$
  - **根据卡尔曼增益参数来对估计值和测量值进行权重分配的公式**
    $X_{(k|k)} = X_{(k|k-1)} + K_{(k)}*(Z_{(k)} - Z_{(k-1)}) = X_{(k|k-1)} + K_{(k)}*(Z_{(k)}-H*X_{(k|k-1)})$
  - **协方差矩阵**
    $P_{(k|k)} = E[(X_{(k)} - X_{(k|k)}) (X_{(k)} - X_{(k|k)})^T]$
## 公式变量统一讲解
  因为卡尔曼滤波的算法公式有各种各样的写法，然后数学符号多样化，给理解带来了一些混乱，我把目前网络上常见的公式数学符号进行了统一梳理和解释。
  - **转移系数矩阵**
    表示状态转移系数矩阵，n×n 阶
    $A$
  - ***控制输入增益矩阵(有的公式可以没有，也可以有)**
    表示可选的控制输入的增益矩阵
    $B$
  - **在网上常见的公式推到中，预测值的表示**
    $X(k|k-1)$,依赖于上一次的值，就是预测值
    $x^-$或者$x^{'}$表示预测值
    $x^-$或者$x^{'}$ 和 $X_{(k-1|k-1)}$ 都是表示预测值
  - **估计值的表示**
    $X(k|k)$,这个为估计值，估计值已经是最接近真实值的值，真实值无法得到
  - **各种噪声的表示**
    - 过程噪声
      $Q$:表示过程激励噪声的协方差矩阵
      状态预测的噪声，过程噪声，符合正态分布$N(0,Q)$
      $W$

    - 观测噪声
      测量的噪声，符合正态分布$N(0,R)$
      $V$
      $v_{k}$
  - **协方差讲解**
    $K_{k}$卡尔曼增益
    
    $P_k$协方差矩阵
    
  - **控制输入**
    $u(k-1)$:控制输入
  

# 公式推导
## 公式推导前的讲解
  - 两个基本问题：  
    1. 卡尔曼滤波算法要做什么？
      对状态进行估计。  
    2. 卡尔曼滤波算法怎么对状态进行估计？      
      利用状态过程噪声和测量噪声对状态进行估计。
    
    一个状态在一个时刻点k的状态进入下一个时刻点k+1状态，会有很多外界因素的干扰，我们把干扰就叫做过程噪声，（这个词一看就是硬翻译过来的，别在意为什么叫噪声）用w表示。任何一个测量仪器，都会有误差，我们把这个误差叫做量测噪声，用v表示。
    回到上面那个公式，状态方程表示状态在不断的更新，从一个时刻点进入下一个时刻点，这个很好理解。关键是量测方程，它表示，我们不断更新的状态有几个能用测量仪器测出来，比如，汽车运动状态参数有很多，比如速度，轮速，滑移率等，但是我们只能测量出轮速，因此量测方程要做的就是把状态参数中能量测的状态拿出来。   
    我们始终要记得我们要做的事：我们要得到的是优化的状态量Xk。   理解了上面之后就可以开始推导公式了。
  - 首先不考虑过程噪声对状态进行更新，很简单：
    $\hat{x}_{k}^{'} = A_k \hat{x}_{(k-1)}$
  - 不考虑测量噪声取出能测量的状态，也很简单：
    $\hat{y}_{k}^{'} = C_k \hat{x}_{k}^{'} = C_k A_k \hat{x}_{(k-1)}$
  - 用测量仪器测量出来的状态值（大家可以考虑到：测量的值就是被各种噪声干扰后的真实值）减去上面不考虑噪声得到的测量值
    $ \tilde{y}_{k} = y_k - \hat{y}_{k}^{'}$

    这个值在数学上是一个定义值，叫做新息，有很多有趣的性质，感兴趣的可以自己谷歌。
    我们对步骤暂且停一停。这个叫新息的值有什么用？由上面的过程我们可以明显看到，它反映了过程噪声和测量噪声综合对测量状态值的影响，也就是它包含了w和v的情况。
  - 一个数值c由两部分内容a和b组成，那么怎样用数学表达式来表达？
    一般有两种做法：
    I.直接相加：c=a+b;
    II. 用比例的方法：a=n*c,b=(1-n)*c

    卡尔曼采用了方法II，用比例的方法来做（其实这也是为什么叫做滤波的原因，因为滤波就是给权值之类的操作）。也就是说，过程噪声w=新息*一个比例。这样得到的过程噪声加上原来（第一步）不考虑过程噪声的状态值不就是优化值了吗？ 也就是：

    $\hat{x}_{k}^{'} = A_k \hat{x}_{(k-1)} + H_k\tilde{y}_{k}$
    $ = A_k \hat{x}_{(k-1)} + H_k(y_k - \hat{y}_{k}^{'})$
    $ = A_k \hat{x}_{(k-1)} + H_k(y_k - C_k A_k \hat{x}_{(k-1)})$

    $H_k$ 卡尔曼增益
    $A_k$ $C_k$ $w_k$的协方差为$Q$
    $v_k$的协方差为$R$
    系统协方差初始值$P_0$,状态初始值$X_0$，都已知。为什么已知，你实际做项目就知道了。
  - 理解公式
    我们可以从一阶低通滤波算法中得到启示
    $ Y(n) = \alpha X(n) * (1-\alpha ) * Y(n - 1)$
    式中：
    $\alpha$:滤波系数；
    X(n):本次采样值
    Y(n-1)上次滤波输出值
    Y(n)=本次滤波输出值。
    这个$\alpha$值就是一个系数，来决定多少权重给上一次输出的值
    卡尔曼滤波中的$H_k$就跟这个系数很相似，分配多少权重给过程值和观测值