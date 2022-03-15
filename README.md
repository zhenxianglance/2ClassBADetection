# Post-Training Detection of Backdoor Attacks for Two-Class and Multi-Attack Scenarios (ICLR2022)

## Background

Backdoor attack is an important type of adversarial attack against deep neural network (DNN) classifiers.
A classifier being attacked will predict to the attacker's target class when a test sample is embedded with a backdoor pattern/trigger.
Backdoor defenses can be deployed post-training, where the defender is, e.g. a downstream user of an app with a DNN classifier, who wants to know if the classifier has been attacked or not.
Such a post-training backdoor defense scenario is very challenging, since: 1) the defender has no access to the classifier's training set, and 2) there are no clean classifiers for reference (e.g. to set a detection threshold).

Reverse-engineering-based defense (RED) is an important family of post-training backdoor defense.
Typically, a RED reverse-engineers the backdoor pattern (e.g.) for each class pair.
Then, an unsupervised anomaly detector is built using the statistics obtained from the estimated backdoor patterns, (e.g.) one statistic per class pair.
An attack is detected if there exists an atypical statistic.
However, when there are only two classes in the classification domain, there will be insufficient statistics to reliably estimate a null distribution for assessing atypicality.
Thus, REDs are generally not applicable to two-class scenarios (or when the number of classes is not much larger than the number of attacks).

## Method

We process each class independently by obtaining an expected transferability (ET) statistic for each class. Thus, there is no need to estimated a null distribution, and the method is applicable to two-class scenarios.
In our paper, we show that there is a constant threshold on ET (i.e. 1/2) for distinguishing backdoor target classes from non-target classes.
Thus, like RED, we don't need supervision to set a detection threshold.

## How to use

This repository contains code that can be used to evaluate our ET framework on a variety of datasets and for multiple different backdoor patterns.
The configurations for: 1) create an attack, 2) train a classifier, and 3) perform detection are all specified in `config.json`.
In particular, the choices of datasets are: "cifar10", "cifar100", "stl10", "fmnist", "mnist".
The choices of the DNN architectures include: "resnet18", "vgg11", and "lenet5".
The types of backdoor pattern include: "perturbation" and "patch".
Perturbation shapes include: "pixel", "cross", "square", "chessboard", "X", and "static".
The perturbation size for each shape can be found in the paper.
Patch types include: "noise" and "uniform".
For CIFAR-10 and CIFAR-100, the patch size can be set to 3x3. But for STL-10 with higher resolution, the patch size should be larger, e.g. 10x10. Also for STL-10 and patch backdoor, "noise" patch is easier to be learned than "uniform" patch
"NUM_IMG_WARMUP" is the number of images used to find a good initial patch for patch replacement patterns.
This step is complementary to the patch reverse-engineering method of Neural Cleanse to find a patch related with the true backdoor patch.
For all the classifier being attacker, one should first make sure that the attack is successful before applying a defense.
For MNIST and F-MNIST, the attack may fail when the perturbation/patch overlaps with the foreground object (usually located in at the center).

To run an experiment, we first create an attack (with the configurations specified):

    python attack.py

Then we can either train a classifier being attacked:

    python train_contam.py

or, train a clean classifier

    python train_clean.py

Finally, we detect the attack:

    python detection.py

    
## Citation
If you find our work useful in your research, please consider citing:

	@InProceedings{xiang2022BP2Class,
	  title={Post-Training Detection of Backdoor Attacks for Two-Class and Multi-Attack Scenarios},
	  author={Xiang, Zhen and Miller, David J and Kesidis, George},
	  booktitle = {ICLR},
	  year={2022}
	}