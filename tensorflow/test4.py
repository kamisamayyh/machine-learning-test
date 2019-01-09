import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#      title:tf session
# matrix1 = tf.constant([[3,3]])
# matrix2 = tf.constant([[2],[2]])
#
# product = tf.matmul(matrix1,matrix2)
#
# # method 1
# sess = tf.Session()
# result = sess.run(product)
# print(result)
# sess.close()
#
# # method 2
# with tf.Session() as sess:
#     result2 = sess.run(product)
#     print(result2)



#        title:tf variable
# state = tf.Variable(0,name='counter')
# one = tf.constant(1)
# new_value = tf.add(state,one)
# update = tf.assign(state,new_value)
#
# init = tf.initialize_all_variables()
#
# with tf.Session() as sess:
#     sess.run(init)
#     for _ in range(3):
#         sess.run(update)
#         print(sess.run(state))

#    title:tf placeholder
# input1 = tf.placeholder(tf.float32)
# input2 = tf.placeholder(tf.float32)
#
# output = tf.multiply(input1,input2)
#
# with tf.Session() as sess:
#     print(sess.run(output,feed_dict={input1:[7.],input2:[2.]}))# input feed_dict in there

#     title: def add_layer
def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b = tf.matmul(inputs,Weights) +biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data) - 0.5 +noise

xs = tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[None,1])

l1 = add_layer(xs,1,10,activation_function=tf.nn.relu)
predition = add_layer(l1,10,1,activation_function=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-predition),
                     reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

fig= plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion()
plt.show()
for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i%50:
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass

        predition_value = sess.run(predition,feed_dict={xs:x_data})
        lines = ax.plot(x_data,predition_value,"r-",lw=5)
        plt.pause(0.1)
       #print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))