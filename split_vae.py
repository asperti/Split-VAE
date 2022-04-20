import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input,Conv2D, Conv2DTranspose, Dense, Reshape, \
    BatchNormalization, GlobalAveragePooling2D, Flatten, Activation
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import plotting_github
import fid
from load_data import load_cifar10,load_celeba,load_mnist
from sklearn.mixture import GaussianMixture
import os
import time

#ResNet Blocks

def ResBlock(out_dim, depth=2, kernel_size=3, name='ResBlock'):
    def body(inputs, **kwargs):
      with K.name_scope(name):
        y = inputs
        for i in range(depth):
            y = BatchNormalization(momentum=.999,epsilon=1e-5)(y)
            y = Activation('swish')(y) #ReLU()(y)
            y = Conv2D(out_dim,kernel_size,padding='same')(y)
        s = Conv2D(out_dim, kernel_size,padding='same')(inputs)
      return y + s
    return(body)

def ResFcBlock(out_dim, depth=2, name='ResFcBlock'):
    def body(inputs, **kwargs):
      with K.name_scope(name):
        y = inputs
        for i in range(depth):
            y = BatchNormalization(momentum=.999,epsilon=1e-5)(y)
            y = Activation('swish')(y) #ReLU()(y)
            y = Dense(out_dim)(y)
        s = Dense(out_dim)(inputs)
      return y + s
    return(body)

def ScaleBlock(out_dim, block_per_scale=1, depth_per_block=2, kernel_size=3, name='ScaleBlock'):
    def body(inputs, **kwargs):
      with K.name_scope(name):
        y = inputs
        for i in range(block_per_scale):
            y = ResBlock(out_dim,depth=depth_per_block, kernel_size=kernel_size)(y)
      return y
    return (body)

def ScaleFcBlock(out_dim, block_per_scale=1, depth_per_block=2, name='ScaleFcBlock'):
    def body(inputs, **kwargs):
      with K.name_scope(name):
        y = inputs
        for i in range(block_per_scale):
            y = ResFcBlock(out_dim, depth=depth_per_block)(y)
      return y
    return(body)

# Model

def Encoder(input_shape, base_dim, kernel_size, num_scale, block_per_scale, depth_per_block,
             embedding_dim, name='Encoder'):
    with K.name_scope(name):
        dim = base_dim
        enc_input = Input(shape=input_shape)
        y = Conv2D(dim,kernel_size,padding='same',strides=2)(enc_input)
        for i in range(num_scale-1):
            y = ScaleBlock(dim, block_per_scale, depth_per_block, kernel_size)(y)
            if i != num_scale - 1:
                dim *= 2
                y = Conv2D(dim,kernel_size,strides=2,padding='same')(y)

        y = GlobalAveragePooling2D()(y)
        ySB = ScaleFcBlock(embedding_dim,1,depth_per_block)(y)
        
        encoder = Model(enc_input,ySB)
    return encoder

def Latent(embedding_dim,latent_dim, name="Latent"):
    with K.name_scope(name):
        emb = Input(shape=embedding_dim)
  
        mu_z = Dense(latent_dim)(emb)
        logvar_z = Dense(latent_dim)(emb)
        z = mu_z + K.random_normal(shape=K.shape(mu_z)) * K.exp(logvar_z*.5)

        back_to_emb = Dense(embedding_dim)
        emb_hat = back_to_emb(z)

        noise = Input(shape=latent_dim)
        emb_gen = back_to_emb(noise)
        
        through_latent = Model(emb,[emb_hat,z,mu_z,logvar_z])
        emb_generator = Model(noise,emb_gen)
        return through_latent,emb_generator

def Decoder(out_ch, embedding_dim, dims, scales, kernel_size, block_per_scale, depth_per_block, name='Decoder'):
  
    base_wh = 4
    data_depth = out_ch
    print("dims[0] is = ",dims[0])
    print("embedding_dim is ",embedding_dim)

    with K.name_scope(name):
        emb = Input(shape=(embedding_dim,))
        y = Dense(base_wh * base_wh * dims[0])(emb)
        y = Reshape((base_wh,base_wh,dims[0]))(y)

        for i in range(len(scales) - 1):
            y = Conv2DTranspose(dims[i+1], kernel_size, strides=2, padding='same')(y)
            y = ScaleBlock(dims[i+1],block_per_scale, depth_per_block, kernel_size)(y)

        x_hat = Conv2D(data_depth, kernel_size, 1, padding='same', activation='sigmoid')(y)
        decoder = Model(emb,x_hat)
    return(decoder)

def FullModel(input_shape,latent_dim,base_dim=32,emb_dim=512, kernel_size=3,num_scale=3,block_per_scale=1,depth_per_block=2):
    desired_scale = input_shape[1]
    scales, dims = [], []
    current_scale, current_dim = 4, base_dim
    while current_scale <= desired_scale:
        scales.append(current_scale)
        dims.append(current_dim)
        current_scale *= 2
        current_dim = min(current_dim * 2, 1024)
    assert (scales[-1] == desired_scale)
    dims = list(reversed(dims))
    print(dims,scales)

    encoder = Encoder(input_shape, base_dim, kernel_size, num_scale, block_per_scale, depth_per_block, emb_dim)
    through_latent,emb_generator = Latent(emb_dim,latent_dim)
    
    decoder = Decoder(input_shape[2]*2+1, emb_dim, dims, scales, kernel_size, block_per_scale, depth_per_block)
    

    x = Input(shape=input_shape)
    channels = input_shape[2]
    gamma = Input(shape=())
    emb = encoder(x)
    emb_hat, z, z_mean, z_log_var = through_latent(emb)
    dec = decoder(emb_hat)
    mask = dec[:,:,:,0:1]
    img1 = dec[:,:,:,1:1+channels]
    img2 = dec[:,:,:,1+channels:1+2*channels]
    x_hat = img1*mask + img2*(1-mask)

    vae = Model([x,gamma],x_hat)
    
    #loss
    L_rec =.5 * K.sum(K.square(x-x_hat), axis=[1,2,3]) / gamma 
    L_KL = .5 * K.sum(K.square(z_mean) + K.exp(z_log_var) - 1 - z_log_var, axis=-1)
    L_tot = K.mean(L_rec + beta * L_KL) 

    vae.add_loss(L_tot)

    return(vae,encoder,decoder,through_latent,emb_generator)


####################################################################
# main
####################################################################

dataset = 'celeba' #'mnist','cifar10','celeba'

if dataset == 'celeba':
  latent_dim = 150
  input_dim = (64,64,3)
  beta = 3
  x_train, x_test = load_celeba(data_folder = '/home/andrea/CELEBA/data')
  GMM_components = 100
elif dataset == 'cifar10':
  latent_dim = 200
  input_dim = (32,32,3)
  beta = 3
  x_train, x_test = load_cifar10()
  GMM_components = 100
elif dataset == 'mnist':
  latent_dim = 32
  input_dim = (32,32,1)
  beta = 8
  x_train, x_test = load_mnist()
  GMM_components = 100
  
vae,encoder,decoder,through_latent,emb_generator = FullModel(input_dim,latent_dim,base_dim=64,num_scale=4,emb_dim=512)

#vae.summary()

batch_size = 100
initial_lr = .0001
optimizer = Adam(learning_rate=initial_lr)

vae.compile(optimizer=optimizer,loss=None,metrics=['mse'])
epochs = 0
load_w = True

weights_filename = 'my_weights' #visit    for pre-trained weights

if load_w:
    vae.load_weights('weights/'+weights_filename)
    xin = x_train[:10000]
    #estimate current mse for intializing gamma
    #the value of gamma is irrelevant for prediction
    gamma_in = np.ones(10000)
    xout = vae.predict([xin,gamma_in])
    mseloss = np.mean(np.square(xin - xout))
    print("mse = {0:.5f}".format(mseloss))
else:
    mseloss = 1.

gamma = mseloss
gamma_in = np.zeros(batch_size)

num_sample = np.shape(x_train)[0]
print('Num Sample = {}.'.format(num_sample))
iteration_per_epoch = num_sample // batch_size

for epoch in range(epochs):
    np.random.shuffle(x_train)
    epoch_loss = 0
    for j in range(iteration_per_epoch):
        image_batch = x_train[j*batch_size:(j+1)*batch_size]
        gamma_in[:] = gamma #mseloss

        loss, bmseloss = vae.train_on_batch([image_batch, gamma_in],image_batch)
        epoch_loss += loss
        #we estimate mse as a weighted combination of the
        #the previous estimation and the minibatch mse
        #mseloss = min(mseloss,mseloss*.99+bmseloss*.01)
        #see https://ieeexplore.ieee.org/document/9244048 for details
        mseloss = mseloss * .99 + bmseloss * .01
        gamma = min(mseloss, gamma)
        if j % 50 == 0:
            #print("mse: ", mseloss)
            print(".", end='',flush=True)
    epoch_loss /= iteration_per_epoch
    print('Date: {date}\t'
          'Epoch: [Stage 1][{0}/{1}]\t'
          'Loss: {2:.4f}.'.format(epoch+1, epochs, epoch_loss,
                                  date=time.strftime('%Y-%m-%d %H:%M:%S')))
    print("mse: ", mseloss)
    if True: #save weights after each epoch
        vae.save_weights('weights/' + weights_filename)

def get_stats():
    np.set_printoptions(precision=3,suppress=True)
    emb = encoder.predict(x_test)
    emb_hat, z, z_mean, z_log_var = to_latent(emb)
    print("variance law: ", np.var(z_mean, axis=0) + np.mean(np.exp(z_log_var),axis=0))
    print("(mean) variance of means = ",np.mean(np.var(z_mean, axis=0)))
    print("mean of variances: ", np.mean(np.exp(z_log_var)))

    #check for inactive variables
    #we consider a variable inactive if its variance is below .05
    print("inactive variables =", np.sum(np.var(z_mean, axis=0)<.05))
    #the computed log_var should be close to 1
    print("inactive variables (2) =", np.sum(np.mean(np.exp(z_log_var), axis=0)>.95))

    
def compute_fid(REC=False,GEN=False,GMM=True,X=True,X1=True,X2=True,MIX=True):
    true_images = x_test[0:10000]
    channels = true_images.shape[3]
    if REC: #to compute FID for reconstructed images
        emb = encoder.predict(x_test)
        emb_hat, z, z_mean, z_log_var = through_latent(emb)
        dec = decoder.predict(emb_hat,batch_size=100)
        mask = dec[:,:,:,0:1]
        img1 = dec[:,:,:,1:1+channels]
        img2 = dec[:,:,:,1+channels:1+2*channels]
        generated = img1*mask + img2*(1-mask)
        fidscore = fid.get_fid(true_images, generated)
        print("reconstructed = ", fidscore)
    if GEN: #to compute FID for generated images (no GMM)
        seed = np.random.normal(0,1,(10000,latent_dim))
        emb = emb_generator(seed)
        dec = decoder.predict(emb,batch_size=50)
        plot_three(dec)
        mask = dec[:,:,:,0:1]
        img1 = dec[:,:,:,1:1+channels]
        img2 = dec[:,:,:,1+channels:1+2*channels]
        generated = img1*mask + img2*(1-mask)
        fidscore = fid.get_fid(true_images,x_train[0:10000])
        print("real = ", fidscore)
        fidscore = fid.get_fid(true_images,generated)
        print("generated 1 = ", fidscore)
        fidscore = fid.get_fid(true_images,img1)
        print("img 1 = ", fidscore)
        fidscore = fid.get_fid(true_images,img2)
        print("img 2 = ", fidscore)
    if GMM: #to apply GMM 
        emb = encoder.predict(true_images)
        emb_hat, z, z_mean, z_log_var = through_latent(emb)
        z_density = GaussianMixture(n_components=GMM_components, max_iter=200) 
        print("Fitting GMM")
        z_density.fit(z_mean)
        seed,_ = z_density.sample(n_samples=10000)
        emb = emb_generator(seed)
        dec = decoder.predict(emb,batch_size=50)
        #plot_three(dec)
        mask = dec[:,:,:,0:1]
        img1 = dec[:,:,:,1:1+channels]
        img2 = dec[:,:,:,1+channels:1+2*channels]
        generated = img1*mask + img2*(1-mask)
        img_mix =np.copy(img1)
        img_mix[5000:] = img2[5000:]
        plotfig = True
        while plotfig:
          plotting_github.show_fig_square(mask[np.random.randint(0,10000,100)])
          a = input("q to quit")
          if a == "q":
            plotfig = False
        if X:
          fidscore = fid.get_fid(true_images, generated)
          print("generated GMM = ", fidscore)
        if X1:
          fidscore = fid.get_fid(true_images, img1)
          print("img1 GMM = ", fidscore)
        if X2:
          fidscore = fid.get_fid(true_images, img2)
          print("img2 GMM = ", fidscore)
        if MIX:
          fidscore = fid.get_fid(true_images, img_mix) 
          print("mix generated GMM = ", fidscore)

compute_fid(REC=False,GEN=False,GMM=True,X=True,X1=True,X2=True,MIX=True)
