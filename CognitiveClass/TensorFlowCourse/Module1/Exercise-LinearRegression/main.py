import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("PierceCricketData.csv")
df.head()

x_data, y_data = (df["Chirps"].values,df["Temp"].values)
'''
plt.plot(x_data, y_data, 'ro')
plt.xlabel("# Chirps per 15 sec")
plt.ylabel("Temp in Farenhiet")
#plt.show()
'''

'''
the model that best fits its something like y = mx+c
'''
X = tf.placeholder(tf.float32, shape=(x_data.size))
Y = tf.placeholder(tf.float32, shape=(y_data.size))
m= tf.Variable(3.0)
c = tf.Variable(2.0)

#construct the model
Ypred = tf.add(tf.multiply(X,m),c)
session = tf.Session()
session.run(tf.global_variables_initializer())

pred = session.run(Ypred, feed_dict={X:x_data})

plt.plot(x_data, pred)
plt.plot(x_data, y_data, 'ro')
plt.xlabel("# Chirps per 15 sec")
plt.ylabel("Temp in Farenhiet")
plt.show()

# normalization factor
nf = 1e-1
# seting up the loss function
loss = tf.reduce_mean(tf.squared_difference(Ypred*nf,Y*nf))


optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
#optimizer = tf.train.AdagradOptimizer(0.01 )
# pass the loss function that optimizer should optimize on.
train = optimizer.minimize(loss)
session.run(tf.global_variables_initializer())

##train to predit 'm' and 'c'

convergenceTolerance = 0.0001
previous_m = np.inf
previous_c = np.inf

steps = {}
steps['m'] = []
steps['c'] = []

losses = []

for k in range(100000):
    ########## Your Code goes Here ###########

    # run a session to train , get m and c values with loss function
    _, _m, _c, _l = session.run([train, m, c, loss], feed_dict={X: x_data, Y: y_data})


    steps['m'].append(_m)
    steps['c'].append(_c)
    losses.append(_l)
    if (np.abs(previous_m - _m) <= convergenceTolerance) or (np.abs(previous_c - _c) <= convergenceTolerance):
        print("Finished by Convergence Criterion")
        print(k)
        print(_l)
        break
    previous_m = _m,
    previous_c = _c,

session.close()

plt.plot(steps['m'])
plt.show()