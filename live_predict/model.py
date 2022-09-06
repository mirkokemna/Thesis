import numpy as np
import tensorflow as tf


### ------------- Define Model ------------- ###

## define downsampling
def downsample(Nfilters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.1)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(Nfilters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())
  result.add(tf.keras.layers.LeakyReLU())

  return result

## define upsampling
def upsample(filters, size, crop=(0,0)):
  initializer = tf.random_normal_initializer(0., 0.1)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))
  result.add(
  tf.keras.layers.Cropping2D(cropping=((0, crop[0]), (0, crop[1]))))
  result.add(tf.keras.layers.BatchNormalization())
  result.add(tf.keras.layers.ReLU())

  return result



## define generator
output_channels = 2

def Generator():
  inputs = tf.keras.layers.Input(shape=[85, 256, 1])

  down_stack = [
    downsample(64, 4, apply_batchnorm=False),  # (batch_size, 43, 128, 64)
    downsample(128, 4),  # (batch_size, 22, 64, 128)
    downsample(256, 4),  # (batch_size, 11, 32, 256)
    downsample(512, 4),  # (batch_size, 6, 16, 512)
    downsample(512, 4),  # (batch_size, 3, 8, 512)
    downsample(512, 4),  # (batch_size, 2, 4, 512)
    downsample(512, 4),  # (batch_size, 1, 2, 512)
  ]

  up_stack = [
    upsample(512, 4),              # (batch_size, 2, 4, 1024)
    upsample(512, 4, crop=(1,0)),  # (batch_size, 3, 8, 1024)
    upsample(512, 4),              # (batch_size, 6, 16, 1024)
    upsample(512, 4, crop=(1,0)),  # (batch_size, 11, 32, 1024)
    upsample(256, 4),              # (batch_size, 22, 64, 512)
    upsample(128, 4, crop=(1,0)),  # (batch_size, 43, 128, 256)
  ]

  initializer = tf.random_normal_initializer(0., 0.1)
  last = tf.keras.layers.Conv2DTranspose(output_channels, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer)  # (batch_size, 84, 256, 2)

  final_crop = tf.keras.layers.Cropping2D(cropping=((0, 1), (0, 0))) # (batch_size, 83, 256, 2)

  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)
  x = final_crop(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

  return result

generator = Generator()
# generator.summary()



## define discriminator
def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)

  cond = tf.keras.layers.Input(shape=[85, 256, 1], name='input_image')
  field = tf.keras.layers.Input(shape=[85, 256, 2], name='target_image')

  x = tf.keras.layers.concatenate([cond, field])  # (batch_size, 85, 256, 4)

  down1 = downsample(64, 4, False)(x)  # (batch_size, 43, 128, 64)
  down2 = downsample(128, 4)(down1)  # (batch_size, 22, 64, 128)
  down3 = downsample(256, 4)(down2)  # (batch_size, 11, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 13, 34, 256)
  conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1)  # (batch_size, 10, 31, 512)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 12, 33, 512)

  last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2)  # (batch_size, 9, 30, 1)

  return tf.keras.Model(inputs=[cond, field], outputs=last)

discriminator = Discriminator()
# discriminator.summary()





### ------------- Define Training ------------- ###

## Optimizers
generator_optimizer     = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)



def predict(geom, experiment, steps):
  geom = np.expand_dims(geom,[0,3])
  checkpoint.read('/home/mirko/trained_weights/'+experiment+'/checkpoint_'+str(steps))
  prediction = generator(geom)
  # tf.print(tf.shape(geom))
  # tf.print(tf.shape(prediction))
  output = prediction * geom
  return output.numpy()

def judge(case, experiment, steps):
  checkpoint.read('/home/mirko/trained_weights/'+experiment+'/checkpoint_'+str(steps))
  geom          = case[...,0]
  case          = np.expand_dims(case,0)
  geom_ext      = np.expand_dims(geom,[0,-1])
  prediction    = predict(geom, experiment, steps)
  fake_judged   = discriminator([geom_ext,prediction])[...,0]
  truth_judged  = discriminator([geom_ext,case[...,-3:-1]])[...,0]
  output        = np.zeros(np.append(tf.shape(truth_judged),2))
  output[...,0] = fake_judged
  output[...,1] = truth_judged
  return output
