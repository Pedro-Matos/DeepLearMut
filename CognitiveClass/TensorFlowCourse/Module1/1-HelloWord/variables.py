import tensorflow as tf


state = tf.Variable(0)  #variables need to be initialized
one = tf.constant(1)    #normal constant to sum +1
new_value = tf.add(state,one)   #this is an operation to add 1 to the state(variable)
update = tf.assign(state,new_value) #the assign operation that is called on the session
init_op = tf.global_variables_initializer() #to init the variables

with tf.Session() as session:
  session.run(init_op)
  print(session.run(state))
  for _ in range(3):
    session.run(update)
    print(session.run(state))