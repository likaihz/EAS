
import matplotlib.pyplot as plt
import numpy as np


ev=np.loadtxt("results/set_mlp_srelu_sgd_cifar10_acc.txt")
fix=np.loadtxt("results/fixprob_mlp_srelu_sgd_cifar10_acc.txt")
dense=np.loadtxt("results/dense_mlp_srelu_sgd_cifar10_acc.txt")

plt.xlabel("Epochs[#]")
plt.ylabel("CIFAR10\nAccuracy [%]")


plt.plot(dense*100,'b',label="MLP")
plt.plot(fix*100,'y',label="MLP$_{FixProb}$")
plt.plot(ev*100,'r',label="SET-MLP")

plt.legend(loc=4)
plt.grid(True)
plt.tight_layout()
plt.savefig("cifar10_models_performance.pdf")
plt.close()
