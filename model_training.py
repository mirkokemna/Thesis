import numpy as np
import tensorflow as tf
import os
import time
import datetime

# ### ------------- Setup ------------- ###

study_id = os.path.basename(__file__)[:-3]
studypath = 'experiments/' + study_id
os.mkdir(studypath)


# ### ------------- Data Stuff ------------- ###

## import data
train_files = tf.data.Dataset.list_files('samples/training/*', seed=11011997)



def read_case(item):
  data = np.load(item.decode())
  return data.astype(np.float32)[...,:3]

train_set = train_files.map(
  lambda item: tuple(tf.numpy_function(read_case, [item], [tf.float32,])))

batch_size = 32
train_set = train_set.batch(batch_size)




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

## define generator loss
Lambda_L2  = 0
Lambda_L1  = 0
Lambda_discriminator = 1

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(discriminated, gen_output, target):

  discriminator_loss = loss_object(tf.ones_like(discriminated), discriminated)

  pixcount = 85*256
  diff = target - gen_output
  dims = diff.get_shape().as_list()
  dims[-3] = pixcount
  del dims[-2]
  diff = tf.reshape(diff, dims) # reshape from [32 85 256 2] to [32 21760 2]

  l2_err    = tf.norm(diff, ord=2, axis=-1)
  l2_loss   = tf.reduce_mean(l2_err)

  l1_err    = tf.norm(diff, ord=1, axis=-1)
  l1_loss   = tf.reduce_mean(l1_err)

  total_gen_loss = Lambda_discriminator * discriminator_loss + Lambda_L2 * l2_loss + Lambda_L1 * l1_loss

  return total_gen_loss, discriminator_loss, l2_loss, l1_loss



## define discriminator
def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)

  cond = tf.keras.layers.Input(shape=[85, 256, 1], name='input_image')
  field = tf.keras.layers.Input(shape=[85, 256, 2], name='target_image')

  x = tf.keras.layers.concatenate([cond, field])  # (batch_size, 83, 256, 4)

  down1 = downsample(64, 4, False)(x)  # (batch_size, 42, 128, 64)
  down2 = downsample(128, 4)(down1)  # (batch_size, 21, 64, 128)
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


## define discriminator loss
def discriminator_loss_fcn(real_discriminated, forged_discriminated):
  real_loss = loss_object(tf.ones_like(real_discriminated), real_discriminated)
  forged_loss = loss_object(tf.zeros_like(forged_discriminated), forged_discriminated)

  total_loss = real_loss + forged_loss

  return total_loss






### ------------- Define Training ------------- ###

## Optimizers
generator_optimizer     = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


summary_writer = tf.summary.create_file_writer("logs/" + study_id + '_' + datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))

## Training step
@tf.function
def train_step(input_image, target, step):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as discriminator_tape:
    gen_output = generator(input_image, training=True)
    gen_output = tf.math.multiply(gen_output,input_image)


    discriminated_real_output       = discriminator([input_image, target], training=True)
    discriminated_forged_output     = discriminator([input_image, gen_output], training=True)


    gen_total_loss, gen_discriminator_loss, gen_l2_loss, gen_l1_loss = generator_loss(discriminated_forged_output, gen_output, target)

    discriminator_loss_value    = discriminator_loss_fcn(discriminated_real_output,    discriminated_forged_output)

  generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
  discriminator_gradients   = discriminator_tape.gradient(discriminator_loss_value,
                                               discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients,generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,discriminator.trainable_variables))

  with summary_writer.as_default():
    tf.summary.scalar('Discriminator Loss', discriminator_loss_value, step)

    tf.summary.scalar('Total Generator Loss', gen_total_loss, step)
    tf.summary.scalar('Generator Discriminator Loss', gen_discriminator_loss, step)

    tf.summary.scalar('L2 Loss', gen_l2_loss, step)
    tf.summary.scalar('L1 loss = MAE', gen_l1_loss, step)
      




### ------------- Train Model ------------- ###
def fit(train_set, steps):
  steps = steps + 1
  tf.print("\n\n\n****** Experiment "+study_id+", commencing training ******\n\n\n")
  tf.print("%i steps planned\n" %(steps))
  start = time.time()
  interval = 5000

  for step, fields in train_set.repeat().take(steps).enumerate():
    fields = fields[0]
    input_image = tf.expand_dims(fields[...,0],-1)
    target = fields[...,1:]

    train_step(input_image, target, step)

    # Save (checkpoint) the model every 5k steps
    if step % interval == 0:
      if step != 0:
        tf.print(f'Time for {interval} steps: {time.time()-start:.2f} sec\n')
        tf.print(f"Step: {step}")
        start = time.time()

      nstep = step.numpy()
      checkpoint.write(file_prefix=studypath+'/'+'checkpoint_'+str(nstep))

  tf.print("\n\n\n****** training complete ******\n\n\n")
  return


# # restore to checkpoint:
# restorepath = 'experiments/make_checkpoint/'
# nstep = 50
# checkpoint.read(restorepath+'checkpoint_'+str(nstep))


fit(train_set, steps=50000)
