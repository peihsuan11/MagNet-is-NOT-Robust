## test_attack.py -- sample code to test attack procedure
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import tensorflow as tf
import numpy as np
import time
import random


from setup_cifar import CIFAR, CIFARModel
from setup_mnist import MNIST, MNISTModel
from setup_inception import ImageNet, InceptionModel

from l2_attack import CarliniL2
from l0_attack import CarliniL0
from li_attack import CarliniLi

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def show(img):
    """
    Show MNSIT digits in the console.
    """
    remap = "  .*#"+"#"*100
    img = (img.flatten()+.5)*3
    if len(img) != 784: return
    print("START")
    for i in range(28):
        print("".join([remap[int(round(x))] for x in img[i*28:i*28+28]]))


def generate_data(data, samples, targeted=True, start=0, inception=False):
    """
    Generate the input data to the attack algorithm.

    data: the images to attack
    samples: number of samples to use
    targeted: if true, construct targeted attacks, otherwise untargeted attacks
    start: offset into data to use
    inception: if targeted and inception, randomly sample 100 targets intead of 1000
    """
    random.seed(3)
    inputs = []
    targets = []
    true_ids = []
    idx = np.load("ID.npy")
    print (idx[0:10])
    for i in idx:
        if targeted:
            if inception:
                seq = random.sample(range(1,1001), 10)
            else:
                seq = range(data.test_labels.shape[1])

            for j in seq:
                if (j == np.argmax(data.test_labels[start+i])) and (inception == False):
                    continue
                inputs.append(data.test_data[start+i])
                targets.append(np.eye(data.test_labels.shape[1])[j])
                true_ids.append(start+i)
        else:
            inputs.append(data.test_data[start+i])
            targets.append(data.test_labels[start+i])
            true_ids.append(start+i)


    inputs = np.array(inputs)
    targets = np.array(targets)
    true_ids = np.array(true_ids)

    return inputs, targets, true_ids


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-cf", "--conf", type=int, default=0, help='Set attack confidence for transferability tests')
    #parser.add_argument("-sd", "--seed", type=int, default=3, help='random seed for generate_data')
    args = vars(parser.parse_args())
    print(args)
    #main(args)
    
    with tf.Session() as sess:
        data, model =  MNIST(), MNISTModel("models/mnist", sess)
        attack = CarliniL2(sess, model, batch_size=9, max_iterations=1000, confidence=args['conf'])

        inputs, targets ,true_ids= generate_data(data, samples=1008, targeted=False,
                                        start=0, inception=False)
        print (true_ids[0:10])
        print (true_ids.shape)
        timestart = time.time()
        adv = attack.attack(inputs, targets)
        #print (adv.shape)
        
        np.save('ID/labels_train'+str(args['conf'])+'.npy',targets)
        np.save('ID/L2train'+str(args['conf'])+'.npy',adv)
        np.save('ID/ID_train'+str(args['conf'])+'.npy',true_ids)
        
        print (    args['conf']     )


        timeend = time.time()
        
        print("Took",timeend-timestart,"seconds to run",len(inputs),"samples.")

        '''
        for i in range(len(adv)):
            print("Valid:")
            show(inputs[i])
            print("Adversarial:")
            show(adv[i])
            
            print("Classification:", model.model.predict(adv[i:i+1]))

            print("Total distortion:", np.sum((adv[i]-inputs[i])**2)**.5)
        '''
        print (adv.shape)
        print(args)
