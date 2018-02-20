# MagNet-is-NOT-Robust

MagNet is NOT Robust to Transfer Attacks
===
The experiment of "MagNet is NOT Robust to Transfer Attacks".

ABSTRACT
---
In recent years, defending adversarial perturbations to natural examples in order to build robust machine learning models trained by deep neural networks (DNNs) has become an emerging research ﬁeld in the conjunction of deep learning and security. In particular, MagNet consisting of an adversary detector and a data reformer is by far one of the strongest defenses in the black-box setting, where the attacker aims to craft transferable adversarial examples from an undefended DNN model to bypass a defense module without knowing its existence. MagNet can successfully defend a variety of attacks in DNNs, including the Carlini and Wagner’s transfer attack based on the L2 distortion metric. However, in this paper, under the black-box transfer attack setting we show that adversarial examples crafted based on the L1 distortion metric can easily bypass MagNet and
fool the target DNN image classiﬁers on MNIST and CIFAR-10. We also provide theoretical justiﬁcation on why the considered approach can yield adversarial examples with superior attack transferability and conduct extensive experiments on variants of MagNet to verify its lack of robustness to L1 distortion based transfer attacks. Notably, our results substantially weaken the existing transfer attack assumption of knowing the deployed
defense technique when attacking defended DNNs (i.e., the gray-box setting).