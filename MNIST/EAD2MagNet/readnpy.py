import numpy as np
import os
import pickle

#adv = np.load("/home/user/dingbaw/EAD-Attack-1/ENbe=0/ENtrain40_0.0.npy")
adv = np.load("/home/user/dingbaw/EAD-Attack-1/e-3/ID_train0_0.001.npy")
idx = np.load("/home/user/dingbaw/CIFAR/nn_robust_attacks/ID.npy") #20171210
#idx = np.load("/home/user/dingbaw/CW2MagNet/CW_passd15.npy") #20171210
#idx2 = np.load("/home/user/dingbaw/CW2MagNet/CW_pass/CW_passd15.npy") #20171210

print("CW")
print(idx[0:10])
print (adv[0:10])


'''
def load_obj(name, directory='/home/user/dingbaw/MagNet/attack_data/'):
    if name.endswith(".pkl"): name = name[:-4]
    with open(os.path.join(directory, name + '.pkl'), 'rb') as f:
        return pickle.load(f)
idx = load_obj("example_idx")
#idx2 = load_obj("example_carlini_40.0")
print ("THIS IS MAGNET's pkl")
print (idx.shape)

print ("\n")
#print (idx2[1:2])
adv = np.load("/home/user/dingbaw/nn_robust_attacks/untarget_b20_10000/L2train0.npy")
print ("THIS IS EAD")
print (adv.shape)
'''
'''
adv = np.load("L2/L2_0.npy")
print ("THIS IS EAD")
print (adv.shape)

adv = np.load("labels_train10.0.npy")
adv=np.argmax(adv,axis=1)
print('90')
print (adv)
print (adv.shape)


adv = np.load("labels_train0.npy")
adv=np.argmax(adv,axis=1)
print (adv)
print (adv.shape)

adv = np.load("/home/user/dingbaw/MagNet-master/train/beta=0/L1_train0.npy")
adv2 = np.load("/home/user/dingbaw/MagNet-master/train/beta=0/L1_train10.npy")

#adv=np.argmax(adv,axis=1)
print (adv[1][1]+0.5)
print (adv2[1][1]+0.5)

print (adv.shape)

python3 test_attack.py -sh -tr -cf 5;
for i in {1};do python3 test_attack.py -a L1 -tr -cf 0;python python3 test_attack.py -a L1 -tr -cf 5;python python3 test_attack.py -a L1 -tr -cf 15;python python3 test_attack.py -a L1 -tr -cf 25;python python3 test_attack.py -a L1 -tr -cf 35;done

for i in {1};do python3 test_attack.py -a L1 -tr -cf 10;python python3 test_attack.py -a L1 -tr -cf 20;python python3 test_attack.py -a L1 -tr -cf 30;python python3 test_attack.py -a L1 -tr -cf 40;done


for i in {1};do python3 test_attack.py -a L1 -tr -cf 5;python3 test_attack.py -a L1 -tr -cf 15;python3 test_attack.py -a L1 -tr -cf 25;python3 test_attack.py -a L1 -tr -cf 35;done

for i in {1};do python3 test_attack.py -a L1 -tr -cf 10;python3 test_attack.py -a L1 -tr -cf 30;python3 test_attack.py -a L1 -tr -cf 40;done
#

#for i in {1};do python3 test_attack.py -a L2 -tr -cf 40;python3 test_attack.py -a L2 -tr -cf 10;python3 test_attack.py -a L2 -tr -cf 30;done
for i in {1};do python3 test_attack.py -tr -cf 0;python3 test_attack.py -tr -cf 10;python3 test_attack.py -tr -cf 20;done
for i in {1};do python3 test_attack.py -tr -cf 30;python3 test_attack.py -tr -cf 40;done

for i in {1};do python3 test_attack.py -cf 0;python3 test_attack.py -cf 10;python3 test_attack.py -cf 20;python3 test_attack.py -cf 30;python3 test_attack.py -cf 40;done


#for i in {1};do  python3 test_attack.py -cf 0;python3 test_attack.py -cf 10;python3 test_attack.py -cf 20;python3 test_attack.py -cf 30;python3 test_attack.py -cf 40;done
#for i in {1};do python3 test_attack.py -a L1 -tr -cf 0;python3 test_attack.py -a L1 -tr -cf 10;done
#for i in {1};do  python3 test_attack.py -cf 0;python3 test_attack.py -cf 10;python3 test_attack.py -cf 20;python3 test_attack.py -cf 30;python3 test_attack.py -cf 40;done
'''

#for i in {1};do python3 test_attack.py -cf 0;python3 test_attack.py -cf 10;python3 test_attack.py -cf 20;python3 test_attack.py -cf 30;python3 test_attack.py -cf 40;done
