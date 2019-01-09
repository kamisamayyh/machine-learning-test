import tensorflow as tf
from tensorflow.examples.tutorials.mnist import  input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32,[None,784])
#输入数据的地方，第一个参数是代表输入类型，第二个参数[]代表tensor的shape数据尺寸，None代表不限条数，784代表784个维的向量
W = tf.Variable(tf.zeros([784,10]))
#W shape[784,10] 784是特征的维度 10代表10类 zeros代表初始为0
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x,W)+b)


#信息熵公式 H(x) = E[l(xi)] = E[Log(2,1/p(xi))]=-∑p(xi)log(2,p(xi))(i=1,2,..n)
y_ = tf.placeholder(tf.float32,[None,10])
cross_entropy = tf.reduce_mean( - tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
# 定义一个 placeholder 输入真实的label 用来计算cross_entropy(信息熵)
#reduce_mean求平均值 reduce_sum∑求和

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#GradientDescentOpimizer随机梯度下降 0.5为学习速率
tf.global_variables_initializer().run()
#全局参数初始化并执行run
for i in range(10000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    train_step.run({x:batch_xs,y_:batch_ys})

#批训练 每次100个样本
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
#tf.argmax 从一个tensor中寻找最大的序号，tf.argmax(y,1)求预测盖于最大的那一个
#tf.argmax(y_,1)寻找真实的数组类别,tf.equal判断是否一致
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

print(accuracy.eval({x:mnist.test.images,y_:mnist.test.labels}))


