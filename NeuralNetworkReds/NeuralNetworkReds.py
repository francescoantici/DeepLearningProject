import numpy as np
import tensorflow as tf
from Models.Processing import *
from Models.Losses import *
from NeuralNetworkReds.RedsLoader import RedsLoader
from NeuralNetworkReds.layer_utils import ReflectionPadding2D, res_block
from keras.layers import Input, Activation, Add, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense, Flatten, Lambda
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.models import Model
from SSIM_PIL import compare_ssim
from random import randint
from PIL import Image

class NeuralNetworkReds():
    def __init__(self):
        # the paper defined hyper-parameter:chr
        self.channel_rate = 64
        # Note the image_shape must be multiple of patch_shape
        self.image_shape = (256, 256, 3)
        self.patch_shape = (self.channel_rate, self.channel_rate, 3)

        self.ngf = 64
        self.ndf = 64
        self.input_nc = 3
        self.output_nc = 3
        self.input_shape_generator = (self.image_shape[0], self.image_shape[1], self.input_nc)
        self.input_shape_discriminator = (self.image_shape[0], self.image_shape[1], self.output_nc)
        self.n_blocks_gen = 9

        self._generator = self._generator_model()
        self._discriminator = self._discriminator_model()
        self._g_d = self._generator_containing_discriminator(self._generator_model(), self._discriminator_model())
        self._g_d_m = self._generator_containing_discriminator_multiple_outputs(self._generator, self._discriminator)
        
    def fit(self, datagen, epochs = 1, critic_updates = 5):
        batch_size =  128
        train_size = 21000
        train, _, _ = datagen

        d_opt = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        d_on_g_opt = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)     

        self._discriminator.trainable = True
        self._discriminator.compile(optimizer=d_opt, loss=wasserstein_loss)
        self._discriminator.trainable = False
        loss = [perceptual_loss, wasserstein_loss]
        loss_weights = [100, 1]
        self._g_d_m.compile(optimizer=d_on_g_opt, loss=loss, loss_weights=loss_weights)
        self._discriminator.trainable = True

        output_true_batch, output_false_batch = np.ones((batch_size, 1)), -np.ones((batch_size, 1))
        for epoch in range(epochs):

            d_losses = []
            d_on_g_losses = []
            for index in range(int(train_size / batch_size)):
                x_train = []
                y_train = []
                for i in range(batch_size):
                    d = next(train)
                    x_train.append(preprocess(d[0]))
                    y_train.append(preprocess(d[1]))
                x_train = np.asarray(x_train).reshape(batch_size, self.image_shape[0], self.image_shape[1], 3)
                y_train = np.asarray(y_train).reshape(batch_size, self.image_shape[0], self.image_shape[1], 3)
                batch_indexes = np.random.permutation(batch_size)
                image_blur_batch = x_train[batch_indexes]
                image_full_batch = y_train[batch_indexes]

                generated_images = self._generator.predict(x=image_blur_batch, batch_size=batch_size)

                for _ in range(critic_updates):
                    d_loss_real = self._discriminator.train_on_batch(image_full_batch, output_true_batch)
                    d_loss_fake = self._discriminator.train_on_batch(generated_images, output_false_batch)
                    d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
                    d_losses.append(d_loss)

                self._discriminator.trainable = False

                d_on_g_loss = self._g_d_m.train_on_batch(image_blur_batch, [image_full_batch, output_true_batch])
                d_on_g_losses.append(d_on_g_loss)

                self._discriminator.trainable = True

            print(np.mean(d_losses), np.mean(d_on_g_losses))

    def evaluate(self, datagen, display = True):
        _, _, testgen = datagen
        pictures = 1
        x = []
        y = []
        #Case output is 720x1280
        """
        for i in range(pictures):
                d = next(testgen)
                x.append(patcher(d[0]))
                y.append(patcher(d[1]))
        generated = self._generator.predict(np.asarray(x).reshape(pictures*15, self.image_shape[0], self.image_shape[1], 3))
        y_pred = np.asarray([deprocess_image(elem) for elem in generated])
        y = np.asarray(y)
        """
        #Case 256x256
        for i in range(pictures):
                d = next(testgen)
                x.append(preprocess(d[0]))
                y.append(preprocess(d[1]))
        generated = self._generator.predict(np.asarray(x))
        y_pred = np.asarray([deprocess_image(elem) for elem in generated])
        y = np.asarray(y)
        ssim = np.mean(np.asarray([compare_ssim(Image.fromarray(y[i], 'RGB'), Image.fromarray(y_pred[i], 'RGB')) for i in range(len(y))]))
        print("The loss functions on the test set are MSE: {:.2f}, SSIM: {:.2f}, PSNR: {:.2f}.".format(MSE(y, y_pred), ssim, PSNR(y, y_pred)))
    
    def display_sample(self, datagen):
        _, _, testgen = datagen
        X, y = next(testgen)
        patches = patcher(X)
        #Case 720x1280
        data = [np.asarray(X), np.asarray(y), reconstruct(self._generator.predict(patches))]
        #Case 256x256
        #data = [np.asarray(X.resize((self.image_shape[0], self.image_shape[1]))), np.asarray(y.resize((self.image_shape[0], self.image_shape[1]))), deprocess_image(self._generator.predict(preprocess(X).reshape(1,self.image_shape[0], self.image_shape[1],3)).reshape(self.image_shape))]
        #Case Side by Side
        img = np.concatenate((np.asarray(X.resize((self.image_shape[0], self.image_shape[1]))), deprocess_image(self._generator.predict(preprocess(X).reshape(1,self.image_shape[0], self.image_shape[1],3)).reshape(self.image_shape))), axis = 1)
        data = [img, np.asarray(y.resize((self.image_shape[0], self.image_shape[1])))]
        titles = iter(["Blurred", "Original", "Generated"])
        for batch in data:
            img = Image.fromarray(batch, 'RGB')
            img.show(title = next(titles))

    def _generator_model(self):
        """Build generator architecture."""
        # Current version : ResNet block
        inputs = Input(shape = self.image_shape)

        x = ReflectionPadding2D((3, 3))(inputs)
        x = Conv2D(filters=self.ngf, kernel_size=(7, 7), padding='valid')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            x = Conv2D(filters=self.ngf*mult*2, kernel_size=(3, 3), strides=2, padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

        mult = 2**n_downsampling
        for i in range(self.n_blocks_gen):
            x = res_block(x, self.ngf*mult, use_dropout=True)

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            # x = Conv2DTranspose(filters=int(ngf * mult / 2), kernel_size=(3, 3), strides=2, padding='same')(x)
            x = UpSampling2D()(x)
            x = Conv2D(filters=int(self.ngf * mult / 2), kernel_size=(3, 3), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

        x = ReflectionPadding2D((3, 3))(x)
        x = Conv2D(filters=self.output_nc, kernel_size=(7, 7), padding='valid')(x)
        x = Activation('tanh')(x)

        outputs = Add()([x, inputs])
        # outputs = Lambda(lambda z: K.clip(z, -1, 1))(x)
        outputs = Lambda(lambda z: z/2)(outputs)

        model = Model(inputs=inputs, outputs=outputs, name='Generator')
        return model

    def _discriminator_model(self):
        """Build discriminator architecture."""
        n_layers, use_sigmoid = 3, False
        inputs = Input(shape=self.input_shape_discriminator)

        x = Conv2D(filters=self.ndf, kernel_size=(4, 4), strides=2, padding='same')(inputs)
        x = LeakyReLU(0.2)(x)

        nf_mult, nf_mult_prev = 1, 1
        for n in range(n_layers):
            nf_mult_prev, nf_mult = nf_mult, min(2**n, 8)
            x = Conv2D(filters=self.ndf*nf_mult, kernel_size=(4, 4), strides=2, padding='same')(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(0.2)(x)

        nf_mult_prev, nf_mult = nf_mult, min(2**n_layers, 8)
        x = Conv2D(filters=self.ndf*nf_mult, kernel_size=(4, 4), strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)

        x = Conv2D(filters=1, kernel_size=(4, 4), strides=1, padding='same')(x)
        if use_sigmoid:
            x = Activation('sigmoid')(x)

        x = Flatten()(x)
        x = Dense(1024, activation='tanh')(x)
        x = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=inputs, outputs=x, name='Discriminator')
        return model

    def _generator_containing_discriminator(self, generator, discriminator):
        inputs = Input(shape=self.image_shape)
        generated_image = generator(inputs)
        outputs = discriminator(generated_image)
        model = Model(inputs=inputs, outputs=outputs)
        return model

    def _generator_containing_discriminator_multiple_outputs(self, generator, discriminator):
        inputs = Input(shape=self.image_shape)
        generated_image = generator(inputs)
        outputs = discriminator(generated_image)
        model = Model(inputs=inputs, outputs=[generated_image, outputs])
        return model

    def save(self, file_name):
        self._generator.save(file_name)
    
    def load_weights(self, file_name):
        return self._generator.load_weights(file_name)

    