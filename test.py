import tensorflow as tf

physical_devices = tf.config.list_physical_devices("GPU")
for i in range(len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], True)
strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")

a = tf.ones([1,5,5])
b = tf.ones([1,5,10]) 
with strategy.scope():
    c = tf.matmul(a, b)
    print(c.device)
    print(c)
