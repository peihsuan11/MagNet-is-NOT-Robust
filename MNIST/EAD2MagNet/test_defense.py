## test_defense.py -- test defense
##
## Copyright (C) 2017, Dongyu Meng <zbshfmmm@gmail.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

from setup_mnist import MNIST
from utils import prepare_data
from worker import AEDetector, SimpleReformer, IdReformer, AttackData, Classifier, DBDetector, Operator, Evaluator
import utils
import os
import numpy as np


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

detector_I = AEDetector("./defensive_models/MNIST_I", p=2)
detector_II = AEDetector("./defensive_models/MNIST_II", p=1)
reformer = SimpleReformer("./defensive_models/MNIST_I")

id_reformer = IdReformer()


classifier = Classifier("./models/example_classifier")
detector_JSD = [DBDetector(id_reformer, reformer, classifier, T=10)]
detector_JSD += [DBDetector(id_reformer, reformer, classifier, T=40)]


detector_dict = dict()
detector_dict["I"] = detector_I
detector_dict["II"] = detector_II

for i,det in enumerate(detector_JSD):
    detector_dict["JSD"+str(i)] = det
    print(i)

operator = Operator(MNIST(), classifier, detector_dict, reformer)



idx = np.load("ID.npy") #20171210

idx = idx[0:1000] #select 1000 adv examples



#idx=np.argmax(idx,axis=1)
print(idx.shape)
_, _, Y = prepare_data(MNIST(), idx)

#f = "example_carlini_0.0"
#f="nn_robust_attacks/ID/L2train0.npy"
#f="EAD-Attack-1/e-3/L1train0_0.001.npy"
f="EAD-Attack-EN/e-3/ENtrain0_0.001.npy"

print("A T T A C K I N G")

#testAttack = AttackData(f, Y, "Carlini L2 0.0")
testAttack = AttackData(f, Y, "EAD L1 0.0")

evaluator = Evaluator(operator, testAttack)

#JSD
evaluator.plot_various_confidences("defense_performance",drop_rate={"I": 0.001, "II": 0.001,"JSD0": 0.01,"JSD1":0.01}) #add JSD detector

#NORMAL
#evaluator.plot_various_confidences("defense_performance",drop_rate={"I": 0.001, "II": 0.001})

