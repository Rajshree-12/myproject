#importing the necessary libraries first
#line[8]:
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# line[9]:
np.random.seed(101)
tf.set_random_seed(101)
# Generating random linear data
# There will be 50 data points ranging from 0 to 50
x=np.linspace(0,50,50)
#Adding noise to the random linear data
x+=np.random.uniform(-4,4,50)
#line_1
y=np.linspace(0,50,50)
y+=np.random.normal(1,3,50)
#number of data points
n=len(x)
#line[10]:
#plot of training data
plt.scatter(x,y)
plt.xlabel('x')
plt.ylabel('y')
plt.title("Training Data")
plt.show()
#line[11]:
X=tf.placeholder("float")
Y=tf.placeholder("float")
W=tf.Variable(np.random.randn(), name="W")
b=tf.Variable(np.random.randn(), name="b")
#line_2:
learning_rate=0.01
#line_3:
training_epochs=500
#line_4:
'''declaring the hypothesis line:
    y_pred=X*W+b'''
y_pred=tf.add(tf.multiply(X,W),b)
#line_5:
#declaring the cost function as mean squared error of y_pred and Y as :
cost=tf.reduce_sum(tf.pow(y_pred-Y,2))/(2*n)
#Gradient Descent Optimizer
optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
#Global Variables Initializer()
init=tf.global_variables_initializer()
#Starting the Tensorflow Session
with tf.Session() as sess:
    sess.run(init)
    #Iterating through all the epochs
    for epoch in range(training_epochs):
        #Feeding each data point into the optimizer using Feed Dictionary
        for(_x,_y) in zip(x,y):
            sess.run(optimizer, feed_dict={X:_x,Y:_y})
        #Displaying the result after every 50 epochs
        if(epoch+1)%50==0:
                #Calculating the cost a every epoch
                c=sess.run(cost,feed_dict={X:x,Y:y})
                print("Epoch",(epoch+1),":cost=",c,"W=",sess.run(W),"b=",sess.run(b))


    #Storing necessary values to be used outside the session
    training_cost=sess.run(cost,feed_dict={X:x,Y:y})
    weight=sess.run(W)
    bias=sess.run(b)
#line[13]:
predictions=weight*x+bias
print("Training cost=",training_cost,"Weight=",weight,"bias=",bias,'\n')
#line[14]:
#Plotting the results
plt.plot(x,y,'ro',label='original data')
plt.plot(x,predictions,label='Fitted line')
plt.title('Linear Regression Result')
plt.legend()
plt.show()
#finish
