import tensorflow as tf

##Opg. 1 (Number of params) --------------------------------------------
my_mod = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64,activation = 'sigmoid', input_shape = (1,)),
        tf.keras.layers.Dense(1, activation='linear')])

print(my_mod.summary())
# Første lag er 128 params. Fordi input er (input+bias) * 64
# Andet lag er 65. #64 outputnodes og bias


## Opg. 2 (adding layer) --------------------------------------------
my_mod2 = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64,activation = 'sigmoid', input_shape = (1,)),
        tf.keras.layers.Dense(32,activation = 'sigmoid'),
        tf.keras.layers.Dense(1, activation='linear')])
print(my_mod2.summary()) #65*32 input i det midterste lag

   
## Opg. 3 (functional) --------------------------------------------

inputs = tf.keras.Input(shape = (1,))
n1 = tf.keras.layers.Dense(64, activation = 'sigmoid')(inputs)
n2 = tf.keras.layers.Dense(32,activation = 'sigmoid')(n1)
output = tf.keras.layers.Dense(1, activation='linear')(n2)

my_model = tf.keras.Model(inputs=inputs, outputs=output)
my_model.summary()


## Opg. 4 (shortcut) --------------------------------------------
inputs = tf.keras.Input(shape = (1,))
n1 = tf.keras.layers.Dense(64, activation = 'sigmoid')(inputs)
n2 = tf.keras.layers.Dense(32,activation = 'sigmoid')(n1)
conc_layer = tf.keras.layers.concatenate([inputs,n2])
output = tf.keras.layers.Dense(1, activation='linear')(conc_layer)

my_model2 = tf.keras.Model(inputs=inputs, outputs=output)
my_model2.summary()


test = 896 + 128 + 9248 + 128 + 18496 + 256 + 36928 + 256 + 73856 + 512 + 147584 + 512 + 20490
test
(128+128+256+256+512+512)/2
