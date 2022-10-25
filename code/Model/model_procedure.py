"""
######################################################
# Model Procedure
# by Arda Mavi
# The University of Arizona
# Department of Aerospace and Mechanical Engineering
######################################################

#### Module: "model_procedure.py"

#### Description: Create, load, save model.

#### Functions:
#---------------------------------------------------------

Func: create_model(dt=0.002)

Args:
    - dt (Float):
        Default: 0.002
        The dt value used to create the dataset.

Returns:
    - Model (TensorFlow Object):
        TensorFlow Model

#---------------------------------------------------------

Func: get_inital_states(time_steps=1, h_shape=(1,16,16,64), c_shape=(1,16,16,64))

Args:
    - time_steps (Int):
        Default: 1
        Number of time steps of initial states (h and c).
    - h_shape, c_shape (Tuple):
        Default: h_shape=(16, 16, 64), c_shape=(16, 16, 64)
        Size of the initial h and c states of the ConvLSTM layer in the model
        with shape [height, width, total_channels]

Returns:
    - h, c (TensorFlow Tensors):
        Two TensorFlow Tensor shaped like [time, height, width, total_channels]
        for initial h and c states of the ConvLSTM layer in the model.
        Created using Normal distribution with mean=0 and varience=1.

#---------------------------------------------------------

Func: predict(Model, initial_conidtion, time_steps=1000, get_latent=False)

Args:
    - Model (Tensorflow Object):
        Tensorflow Model Object
    - initial_conidtion (4D Numpy Array):
        Initial conidtion with shape like (1, height, width, total_channels)
    - time_steps (Int):
        Default: 1000
        Number of time steps to predict.
    - get_latent (Bool):
        Default: False
        If True, the funciton will also return the latent space.

Returns:
    - Outputs (4D Numpy Array(s)):
        Predicted time steps with shape like (<time_steps>, height, width, total_channels)
        If 'get_latent' is True, the funciton will also return the predicted latent space
        with shape of (<time_steps>, latent_height, latent_width, total_latent_channels).

#---------------------------------------------------------

#### Classes:
#---------------------------------------------------------

Class PhysicLoss(tensorflow.keras.losses.Loss)

+ Class Func: __init__(self, kernel_lap, kernel_x, kernel_y,
                       kernel_t=np.array([-1,0,1]), dx=1./128,
                       dy=1./128, dt=0.002, viscosity=200., padding=(2,2))
Args:
    - kernel_lap, kernel_x, kernel_y, kernel_t (Numpy Array):
        Defaut kernel_t: np.array([-1,0,1])
        PDDO kernels as Numpy arrays with shape (height, width) for
        x, y, lap as space and t as time axis.
    - dx, dy, dt (Float):
        Default: dx=1./128 , dy=1./128 , dt=0.002
        The dx, dy, dt delta values used to create the dataset.
    - viscosity (Float):
        Default: 1./200
        Viscosity coefficient in equation.
    - padding (Tuple):
        Default: (2,2)
        Padding size acording to used boundary condition.

Created Object Returns:
    - Loss Object (TensorFlow Object):
        TensorFlow loss object for calculating the Physical Loss.

+ Class Func: __call__(self, y)
Args:
    - y (Tensor):
        A tensor with shape [time, height, width, channels]

Called Object Returns:
    - Loss Value (TensorFlow Scaler):
        Returns calculated scaler Physical Loss value.

#---------------------------------------------------------

Class LatentLoss(tensorflow.keras.losses.Loss)

+ Class Func: __init__(self, kernel_lap, kernel_x, kernel_y,
                       kernel_t=np.array([-1,0,1]), dx=1./16,
                       dy=1./16, dt=0.002, viscosity=200.,
                       padding=(2,2), latent_dept = 64)
Args:
    - kernel_lap, kernel_x, kernel_y, kernel_t (Numpy Array):
        Defaut kernel_t: np.array([-1,0,1])
        PDDO kernels as Numpy arrays with shape (height, width) for
        x, y, lap as space and t as time axis.
    - dx, dy, dt (Float):
        Default: dx=1./16 , dy=1./16 , dt=0.002
        The dx, dy, dt delta values of the latent space.
    - viscosity (Float):
        Default: 1./200
        Viscosity coefficient in equation.
    - padding (Tuple):
        Default: (2,2)
        Padding size acording to used boundary condition.
    - latent_dept (Int):
        Default: 64
        Number of total channels in latent space.

Created Object Returns:
    - Loss Object (TensorFlow Object):
        TensorFlow loss object for calculating the Latent Loss.

+ Class Func: __call__(self, y)
Args:
    - y (Tensor):
        A tensor with shape [time, height, width, channels]

Called Object Returns:
    - Loss Value (TensorFlow Scaler):
        Returns calculated scaler Latent Loss value.

#---------------------------------------------------------
"""


import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm as ProgressBar
from tensorflow.keras import layers as KL
from tensorflow.keras import activations as KA
from tensorflow.keras.losses import Loss as KLoss


# Set TensorFlow Backend for Channel Last Data Type:
tf.keras.backend.set_image_data_format('channels_last')


# Create Model Architecture:
def create_model(dt = 0.002):

    # Model Inputs:
    # inpt_x: Model input
    # inpt_h: Previous ConvLSTM Output
    # inpt_c: Previous ConvLSTM State Output

    inpt_x = keras.Input(shape=(128, 128, 2))
    inpt_h = keras.Input(shape=(16, 16, 64))
    inpt_c = keras.Input(shape=(16, 16, 64))

    # Weight Regularization:
    RegL2 = tf.keras.regularizers.L2( l2=0.001 )

    x = inpt_x
    # Encoder -----------------------------------------------------------------------------------------------
    # Encoder Layer 1:
    x = CircularPad(padding=(1,1))(x)
    x = KL.Conv2D(filters=8, kernel_size=(4,4), strides=(2,2), use_bias=True, bias_initializer="zeros", kernel_initializer="RandomUniform")(x)
    x = KL.Activation(KA.tanh)(x)

    # Encoder Layer 2:
    x = CircularPad(padding=(1,1))(x)
    x = KL.Conv2D(filters=32, kernel_size=(4,4), strides=(2,2), use_bias=True, bias_initializer="zeros", kernel_initializer="RandomUniform")(x)
    x = KL.Activation(KA.tanh)(x)

    # Encoder Layer 3:
    x = CircularPad(padding=(1,1))(x)
    x = KL.Conv2D(filters=64, kernel_size=(4,4), strides=(2,2), use_bias=True, bias_initializer="zeros", kernel_initializer="RandomUniform", kernel_regularizer=RegL2)(x)
    encoded = KL.Activation(KA.tanh)(x)
    x = encoded

    # ConvLSTM ----------------------------------------------------------------------------------------------
    # ConvLSTM Layer 1:
    x = CircularPad(padding=(1,1))(x)
    h = CircularPad(padding=(1,1))(inpt_h)
    h, c = ConvLSTM2D(filters=64, kernel_size=(3,3), strides=(1, 1))(x, h, inpt_c)
    x = h

    # Decoder ----------------------------------------------------------------------------------------------
    # Decoder Layer 1:
    x = KL.UpSampling2D(size=(2,2), interpolation="nearest")(x)
    x = CircularPad(padding=(1,1))(x)
    x = KL.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), use_bias=True, bias_initializer="zeros", kernel_initializer="RandomUniform", kernel_regularizer=RegL2)(x)
    x = KL.Activation(KA.tanh)(x)

    # Decoder Layer 2:
    x = KL.UpSampling2D(size=(2,2), interpolation="nearest")(x)
    x = CircularPad(padding=(1,1))(x)
    x = KL.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), use_bias=True, bias_initializer="zeros", kernel_initializer="RandomUniform")(x)
    x = KL.Activation(KA.tanh)(x)

    # Decoder Layer 3:
    x = KL.UpSampling2D(size=(2,2), interpolation="nearest")(x)
    x = CircularPad(padding=(1,1))(x)
    x = KL.Conv2D(filters=8, kernel_size=(3,3), strides=(1,1), use_bias=True, bias_initializer="zeros", kernel_initializer="RandomUniform", activity_regularizer=RegL2)(x)
    x = KL.Activation(KA.tanh)(x)

    # Output Layer -----------------------------------------------------------------------------------------
    x = CircularPad(padding=(2,2))(x)
    x = KL.Conv2D(filters=2, kernel_size=(5,5), strides=(1,1), use_bias=False, kernel_initializer="RandomUniform")(x)

    # Residual Connection ----------------------------------------------------------------------------------
    x = ResidualConnection(factor=dt)(inpt_x, x)

    model = keras.Model(inputs=[inpt_x, inpt_h, inpt_c], outputs=[x, h, c, encoded])

    # print(model.summary())

    return model




# Costume Layers:
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# Circular Padding:
class CircularPad(KL.Layer):
    def __init__(self, padding=(1,1)):
        super(CircularPad, self).__init__()

        # Padding shape as (x, y):
        self.padding = padding

    def call(self, x):
        x = tf.concat([x[:,-self.padding[1]:,:,:], x, x[:,0:self.padding[1],:,:]], axis=1)
        x = tf.concat([x[:,:,-self.padding[0]:,:], x, x[:,:,0:self.padding[0],:]], axis=2)
        return x


# -------------------------------------------------------------------------------------------------------------


# ConvLSTM:
class ConvLSTM2D(KL.Layer):
    def __init__(self, filters, kernel_size, strides):
        super(ConvLSTM2D, self).__init__()

        # x: Input , h: Previous Output , c: Previous ConvLSTM State
        # ConvLSTM Parameters:

        # Input Gate Parameters:
        self.input_x = KL.Conv2D(filters=filters, kernel_size=kernel_size,
                                 strides=strides, use_bias=True, bias_initializer="zeros", kernel_initializer="RandomUniform")

        self.input_h = KL.Conv2D(filters=filters, kernel_size=kernel_size,
                                 strides=(1, 1), use_bias=False, kernel_initializer="RandomUniform")

        # Forget Gate Parameters:
        self.forget_x = KL.Conv2D(filters=filters, kernel_size=kernel_size,
                                 strides=strides, use_bias=True, bias_initializer="zeros", kernel_initializer="RandomUniform")

        self.forget_h = KL.Conv2D(filters=filters, kernel_size=kernel_size,
                                 strides=(1, 1), use_bias=False, kernel_initializer="RandomUniform")

        # State Update Gate Parameters:
        self.state_x = KL.Conv2D(filters=filters, kernel_size=kernel_size,
                                 strides=strides, use_bias=True, bias_initializer="zeros", kernel_initializer="RandomUniform")

        self.state_h = KL.Conv2D(filters=filters, kernel_size=kernel_size,
                                 strides=(1, 1), use_bias=False, kernel_initializer="RandomUniform")

        # Output Gate Parameters:
        self.output_x = KL.Conv2D(filters=filters, kernel_size=kernel_size,
                                 strides=strides, use_bias=True, bias_initializer="ones", kernel_initializer="RandomUniform")

        self.output_h = KL.Conv2D(filters=filters, kernel_size=kernel_size,
                                 strides=(1, 1), use_bias=False, kernel_initializer="RandomUniform")

        # Periodic Activation
        self.periodic_func = PeriodicActivation(channels=filters)


    def call(self, x, h, c):
        # x: Input
        # h: Previous Output
        # c: Previous ConvLSTM State

        # N... represents the "New" to point out variables produced in the layer.

        # Input Gate:
        Ni = KA.sigmoid( self.input_x(x) + self.input_h(h) )
        # Forget Gate:
        Nf = KA.sigmoid( self.forget_x(x) + self.forget_h(h) )
        # State Update Gate:
        Nc = self.periodic_func( self.state_x(x) + self.state_h(h) )
        Nc = Nf * c + Ni * Nc
        # Output Gate:
        No = KA.sigmoid( self.output_x(x) + self.output_h(h) )
        Nh = No * KA.tanh(Nc)

        # Return output and state:
        return Nh, Nc


# -------------------------------------------------------------------------------------------------------------


# Periodic Activation:
class PeriodicActivation(KL.Layer):
    def __init__(self, channels, initial_alpha=[-np.pi*2, np.pi*2], **kwargs):
        super(PeriodicActivation, self).__init__(**kwargs)

        initial_alpha = np.random.uniform(initial_alpha[0], initial_alpha[1], (channels,)).astype("float32")
        self.alpha = tf.Variable(initial_value=initial_alpha, trainable=True)

    def call(self, x):
        return x + ( (1/self.alpha) * tf.math.sin(self.alpha*x)**2 )


# -------------------------------------------------------------------------------------------------------------


# Residual Connection:
class ResidualConnection(KL.Layer):
    def __init__(self, factor):
        super(ResidualConnection, self).__init__()
        self.factor = factor

    def call(self, x_old, x_new):
        return x_old + ( self.factor * x_new )


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Costume Loss:
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# Physical Loss:
class PhysicLoss(KLoss):
    def __init__(self, kernel_lap, kernel_x, kernel_y, kernel_t=np.array([-1,0,1]), dx=1./128, dy=1./128, dt=0.002, viscosity=1./200, padding=(2,2)):
        super(PhysicLoss, self).__init__()

        # Viscosity coefficient
        self.viscosity = viscosity

        # Padding:
        self.pad = padding
        self.Padding = CircularPad(padding)

        # PDDO Kernels Operations as Conv Layers:
        # Creates Conv layer, process kernels to be Conv filters
        # with shape (time, high, width, channel), and puts filters into Conv layers.

        # Kernel Lap:
        self.lap_opt = KL.Conv2D(filters=2, kernel_size=kernel_lap.shape, strides=(1,1), use_bias=False)
        self.lap_opt.build((None, None, None, 2))
        kernel_lap = np.expand_dims(kernel_lap/(dx*dy) , axis=(-1, -2))
        k_1 = tf.concat([kernel_lap, np.zeros_like(kernel_lap)], axis=-1)
        k_2 = tf.concat([np.zeros_like(kernel_lap), kernel_lap], axis=-1)
        kernel_lap = tf.concat([k_1, k_2], axis=-2)
        self.lap_opt.set_weights([kernel_lap])
        self.lap_opt.trainable = False

        # Kernel x:
        self.dx_opt = KL.Conv2D(filters=2, kernel_size=kernel_x.shape, strides=(1,1), use_bias=False)
        self.dx_opt.build((None, None, None, 2))
        kernel_x = np.expand_dims(kernel_x/dx , axis=(-1, -2))
        k_1 = tf.concat([kernel_x, np.zeros_like(kernel_x)], axis=-1)
        k_2 = tf.concat([np.zeros_like(kernel_x), kernel_x], axis=-1)
        kernel_x = tf.concat([k_1, k_2], axis=-2)
        self.dx_opt.set_weights([kernel_x])
        self.dx_opt.trainable = False

        # Kernel y:
        self.dy_opt = KL.Conv2D(filters=2, kernel_size=kernel_y.shape, strides=(1,1), use_bias=False)
        self.dy_opt.build((None, None, None, 2))
        kernel_y = np.expand_dims(kernel_y/dy , axis=(-1, -2))
        k_1 = tf.concat([kernel_y, np.zeros_like(kernel_y)], axis=-1)
        k_2 = tf.concat([np.zeros_like(kernel_y), kernel_y], axis=-1)
        kernel_y = tf.concat([k_1, k_2], axis=-2)
        self.dy_opt.set_weights([kernel_y])
        self.dy_opt.trainable = False

        # Kernel t:
        # Shape: (1, time, high, width, channel)
        self.dt_opt = KL.Conv3D(filters=2, kernel_size=kernel_t.shape+(1, 1), strides=(1, 1, 1), use_bias=False)
        self.dt_opt.build((None, None, None, None, 2))
        kernel_t = np.expand_dims(kernel_t/(dt*2) , axis=(-1, -2, -3, -4))
        k_1 = tf.concat([kernel_t, np.zeros_like(kernel_t)], axis=-1)
        k_2 = tf.concat([np.zeros_like(kernel_t), kernel_t], axis=-1)
        kernel_t = tf.concat([k_1, k_2], axis=-2)
        self.dt_opt.set_weights([kernel_t])
        self.dt_opt.trainable = False


    def __call__(self, y):
        # Input shape: [t, h, w, c]

        # Padding:
        y = self.Padding(y)

        # Padding and Seperating Channels:
        u = y[1:-1,self.pad[0]:-self.pad[0],self.pad[1]:-self.pad[1],:1]
        u = tf.concat([u, u], axis=-1) # Duplicate to use in equation.
        v = y[1:-1,self.pad[0]:-self.pad[0],self.pad[1]:-self.pad[1],1:]
        v = tf.concat([v, v], axis=-1) # Duplicate to use in equation.


        # Applying PDDO kernels:
        # Lap:
        uv_laplace = self.lap_opt(y[1:-1,:,:,:])
        # x:
        uv_x = self.dx_opt(y[1:-1,:,:,:])
        # y:
        uv_y = self.dy_opt(y[1:-1,:,:,:])


        # t:
        y = y[:,self.pad[0]:-self.pad[0],self.pad[1]:-self.pad[1],:]
        y = tf.expand_dims(y, axis=0) # Add a batch dimension (1)
        uv_t = self.dt_opt(y)
        uv_t = tf.squeeze(uv_t, axis=0) # Remove the batch dimension


        # Equation:
        eq_uv = uv_t + u * uv_x + v * uv_y - self.viscosity * uv_laplace


        # Return MSE Loss:
        return tf.math.reduce_mean( tf.math.square( eq_uv ), axis=None, keepdims=False )



# Latent Loss:
class LatentLoss(KLoss):
    def __init__(self, kernel_lap, kernel_x, kernel_y, kernel_t=np.array([-1,0,1]), dx=1./16, dy=1./16, dt=0.002, viscosity=1./200, padding=(2,2), latent_dept = 64):
        super(LatentLoss, self).__init__()

        # Viscosity coefficient
        self.viscosity = viscosity

        # Padding:
        self.pad = padding
        self.Padding = CircularPad(padding)

        # Latent size:
        self.latent_dept = latent_dept

        # PDDO Kernels Operations as Conv Layers:
        # Creates Conv layer, process kernels to be Conv filters
        # with shape (time, high, width, channel), and puts filters into Conv layers.

        # Kernel Lap:
        self.lap_opt = KL.Conv2D(filters=1, kernel_size=kernel_lap.shape, strides=(1,1), use_bias=False)
        self.lap_opt.build((None, None, None, 1))
        kernel_lap = np.expand_dims(kernel_lap/(dx*dy) , axis=(-1, -2))
        self.lap_opt.set_weights([kernel_lap])
        self.lap_opt.trainable = False

        # Kernel x:
        self.dx_opt = KL.Conv2D(filters=1, kernel_size=kernel_x.shape, strides=(1,1), use_bias=False)
        self.dx_opt.build((None, None, None, 1))
        kernel_x = np.expand_dims(kernel_x/dx , axis=(-1, -2))
        self.dx_opt.set_weights([kernel_x])
        self.dx_opt.trainable = False

        # Kernel y:
        self.dy_opt = KL.Conv2D(filters=1, kernel_size=kernel_y.shape, strides=(1,1), use_bias=False)
        self.dy_opt.build((None, None, None, 1))
        kernel_y = np.expand_dims(kernel_y/dy , axis=(-1, -2))
        self.dy_opt.set_weights([kernel_y])
        self.dy_opt.trainable = False

        # Kernel t:
        # Shape: (1, time, high, width, channel)
        self.dt_opt = KL.Conv3D(filters=1, kernel_size=kernel_t.shape+(1, 1), strides=(1, 1, 1), use_bias=False)
        self.dt_opt.build((None, None, None, None, 1))
        kernel_t = np.expand_dims(kernel_t/(dt*2) , axis=(-1, -2, -3, -4))
        self.dt_opt.set_weights([kernel_t])
        self.dt_opt.trainable = False


    def __call__(self, y):
        # Input shape: [t, h, w, c]

        # Padding:
        y = self.Padding(y)

        # Padding and Seperating Channels:
        u = y[1:-1,self.pad[0]:-self.pad[0],self.pad[1]:-self.pad[1],:self.latent_dept//2]
        u = tf.concat([u, u], axis=-1) # Duplicate to use in equation.
        v = y[1:-1,self.pad[0]:-self.pad[0],self.pad[1]:-self.pad[1],self.latent_dept//2:]
        v = tf.concat([v, v], axis=-1) # Duplicate to use in equation.


        # Merge channel and time axises to apply space kernels:
        y_space = y[1:-1,:,:,:]
        y_s_shape = y_space.shape # [t, h, w, c]
        y_space = tf.transpose(y_space, (0,3,1,2)) # [t, c, h, w]
        y_space = tf.reshape(y_space, (y_s_shape[0]*y_s_shape[-1],)+y_s_shape[1:-1]) # [t*c, h, w]
        y_space = tf.expand_dims(y_space, axis=-1) # [t*c, h, w, 1]

        # Applying PDDO kernels:
        # Lap:
        uv_laplace = self.lap_opt(y_space)
        # x:
        uv_x = self.dx_opt(y_space)
        # y:
        uv_y = self.dy_opt(y_space)

        # Seperate channel and time axises:
        y_s_shape = list(y_s_shape)
        y_s_shape[1] -= self.pad[0]
        y_s_shape[2] -= self.pad[1]
        # uv_laplace
        uv_laplace = tf.squeeze(uv_laplace, axis=-1) # [t*c, h, w]
        uv_laplace = tf.reshape(uv_laplace, y_s_shape[:1]+y_s_shape[-1:]+uv_laplace.shape[-2:]) # [t, c, h, w]
        uv_laplace = tf.transpose(uv_laplace, (0,2,3,1)) # [t, h, w, c]
        # x:
        uv_x = tf.squeeze(uv_x, axis=-1) # [t*c, h, w]
        uv_x = tf.reshape(uv_x, y_s_shape[:1]+y_s_shape[-1:]+uv_x.shape[-2:]) # [t, c, h, w]
        uv_x = tf.transpose(uv_x, (0,2,3,1)) # [t, h, w, c]
        # y:
        uv_y = tf.squeeze(uv_y, axis=-1) # [t*c, h, w]
        uv_y = tf.reshape(uv_y, y_s_shape[:1]+y_s_shape[-1:]+uv_y.shape[-2:]) # [t, c, h, w]
        uv_y = tf.transpose(uv_y, (0,2,3,1)) # [t, h, w, c]


        # t:
        y = y[:,self.pad[0]:-self.pad[0],self.pad[1]:-self.pad[1],:] # [t, h, w, c]
        y = tf.transpose(y, (3,0,1,2)) # [c, t, h, w]
        y = tf.expand_dims(y, axis=-1) # Add a channel dimension (1)
        uv_t = self.dt_opt(y)
        uv_t = tf.squeeze(uv_t, axis=-1) # Remove the channel dimension
        uv_t = tf.transpose(uv_t, (1,2,3,0)) # [t, h, w, c]

        # Equation:
        eq_uv = uv_t + u * uv_x + v * uv_y - self.viscosity * uv_laplace


        # Return MSE Loss:
        return tf.math.reduce_mean( tf.math.square( eq_uv ), axis=None, keepdims=False )



# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



# Inital state creator for ConvLSTM:
def get_inital_states(time_steps=1, h_shape=(16,16,64), c_shape=(16,16,64)):
    h = tf.random.normal((time_steps,)+h_shape)
    c = tf.random.normal((time_steps,)+c_shape)
    return h, c



# Predict Function:
def predict(Model, initial_conidtion, time_steps=1000, get_latent=False):
    # Get inital data:
    prev_out = initial_conidtion
    prev_h, prev_c = get_inital_states()

    # Output list:
    outputs = tf.reshape(tf.constant([], dtype=prev_out.dtype), (0,)+prev_out.shape[1:])
    latents = tf.reshape(tf.constant([], dtype=prev_h.dtype), (0,)+prev_h.shape[1:])

    # Predict time series:
    for t in ProgressBar(range(time_steps)):
        prev_out, prev_h, prev_c, encode = Model.predict([prev_out, prev_h, prev_c])
        outputs = tf.concat([outputs, prev_out], axis=0)
        latents = tf.concat([latents, encode], axis=0)

    if get_latent:
        return outputs.numpy(), latents.numpy()
    else:
        return outputs.numpy()



# Main: Testing the model structure:
if __name__ == "__main__":
    # Testing:
    import time
    print("Model and Loss modules control with random data...")
    # Random datas for testing:
    inpt = np.random.uniform(0., 1., (1,128,128,2))
    prev_h = np.random.uniform(0., 1., (1,16,16,64))
    prev_c = np.random.uniform(0., 1., (1,16,16,64))
    kernel = np.random.uniform(0., 1., (5,5))

    # Get Model:
    model = create_model()

    # First input:
    inputs = [inpt, prev_h, prev_c]
    outputs = inpt
    print("Model testing for 3 time step...")
    t1 = time.time()
    for time_step in range(3):
        inpt, prev_h, prev_c = model(inputs)
        inputs = [inpt, prev_h, prev_c]
        outputs = np.concatenate((outputs, inpt), axis=0)
    t2 = time.time()
    print("Data flow duration in the model:", t2-t1)

    # Get Loss:
    calc_loss = PhysicLoss(kernel, kernel, kernel)

    print("Loss testing:")
    t1 = time.time()
    loss = calc_loss(outputs)
    t2 = time.time()
    print("Loss calculation duration:", t2-t1)

    scaler_loss = np.mean(loss)
    print("\nRandom testing loss:", scaler_loss)

    print("Done")
