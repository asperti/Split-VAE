import tensorflow as tf
import os
import numpy as np

"""FID
We need to load the inception model pretrained over imagenet
"""

input_shape = (299,299,3)

from tensorflow.keras.applications.inception_v3 import InceptionV3

# prepare the inception v3 model
model = InceptionV3(include_top=False, pooling='avg', input_shape=input_shape, weights='imagenet')

def grayscale_to_rgb(a):
  b = np.repeat(a,3,axis=3)
  print(b.shape)
  return(b)

def get_inception_activations(inps, batch_size=50):
    n_batches = inps.shape[0] // batch_size

    act = np.zeros([inps.shape[0], 2048], dtype=np.float32)
    for i in range(n_batches):
        inp = inps[i * batch_size:(i + 1) * batch_size]
        inpr = tf.image.resize(inp, (299, 299))
        inpr = inpr*2 - 1 #resize images in the interval [-1.1] (if required)
        act[i * batch_size:(i + 1) * batch_size] = model.predict(inpr, steps=1)

        print('Processed ' + str((i + 1) * batch_size) + ' images.')
    return act

def get_fid(images1, images2):
    from scipy.linalg import sqrtm

    shape = np.shape(images1)[3]
    if shape == 1:
        if np.shape(images1)[1]==32:
          dataset = "mnist32"
        else:
          dataset = "mnist"
        images1 = grayscale_to_rgb(images1)
        images2 = grayscale_to_rgb(images2)
    elif shape == 3 and np.shape(images1)[1]==32:
        dataset = "cifar10"
    else:
        assert (np.shape(images1)[1] == 64)
        dataset = "celeba"
    print("Computing FID for {} images".format(dataset))

    # activation for true images is always the same: we just compute it once
    if os.path.exists(dataset + "_act_mu.npy"):
        mu1 = np.load(dataset + "_act_mu.npy")
        sigma1 = np.load(dataset + "_act_sigma.npy")
    else:
        act1 = get_inception_activations(images1)
        mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
        np.save(dataset + "_act_mu.npy", mu1)
        np.save(dataset + "_act_sigma.npy", sigma1)
    print('Done stage 1 of 2')

    act2 = get_inception_activations(images2)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    print('Done stage 2 of 2')

    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)

    # compute sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    trace = np.trace(sigma1 + sigma2 - 2.0 * covmean)
    print("mean distance = {}, trace = {}".format(ssdiff, trace))
    # calculate score
    fid = ssdiff + trace
    return fid
