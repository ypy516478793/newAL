import matplotlib.pyplot as plt
import h5py

time = "20180831-140309"
acc = "56.23"

resultName = "./results/result_" + time + "_acc_" + acc + ".h5"

with h5py.File(resultName, "r") as f:
    accHist = f['Accuracy'][:]
    lossHist = f['Loss'][:]
    rewardHist = f['rewardHist'][:]
    labelData = f['labelData'][:]
    probs = f['Probs'][:]

plt.hist(probs, bins=50)
plt.show()

print("")