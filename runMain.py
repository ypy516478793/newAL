from CNN_classifier import Classifier
from environment import DataEnv
from AC import Actor, Critic
from config import *
import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import h5py
import os

# set env
env = DataEnv()
s_dim = env.state_dim
a_dim = env.action_dim
clf = Classifier()

sess = tf.Session()

# set RL method
actor = Actor(sess, n_features=s_dim, n_actions=a_dim, lr=LR_A)
critic = Critic(sess, n_features=s_dim, lr=LR_C)

sess.run(tf.global_variables_initializer())

accHist = np.zeros([MAX_EPISODES, env.budgets + 1])
lossHist = np.zeros([MAX_EPISODES, env.budgets + 1])
rewardHist = np.zeros([MAX_EPISODES])
labelData = np.zeros([env.budgets, s_dim])

if OUTPUT_GRAPH:
    tf.summary.FileWriter("./logs/" + LOG + "/RL/", sess.graph)

def save_results(acc, probs):
    if not os.path.exists("./results/"):
        os.makedirs("./results/")
    resultName = "result_" + LOG + "_acc_{0:.2f}".format(acc*100) + ".h5"
    with h5py.File("./results/" + resultName, "w") as f:
        f.create_dataset('Accuracy', data=accHist)
        f.create_dataset('Loss', data=lossHist)
        f.create_dataset('rewardHist', data=rewardHist)
        f.create_dataset('labelData', data=labelData)
        f.create_dataset('Probs', data=probs)

def save(acc):
    saver = tf.train.Saver()
    if not os.path.exists("./RLmodel/"):
        os.makedirs("./RLmodel/")
    saver.save(sess, "./RLmodel/" + "acc_{0:.2f}_epi_{1:d}_budg_{2:d}".format(acc*100, MAX_EPISODES, env.budgets))

def restore(acc, epi, budgets):
    saver = tf.train.Saver()
    saver.restore(sess, "./RLmodel/" + "acc_{0:.2f}_epi_{1:d}_budg_{2:d}".format(acc*100, epi, budgets))

def train():
    # start training
    for i in range(MAX_EPISODES):
        clf.reset()
        state = env.reset(clf)
        ep_r = 0.
        labelNum = 0
        while True:
            action = actor.choose_action(state)
            # store state of labeled sample
            if action == 1:
                labelData[labelNum, :] = state
                labelNum += 1
            state_, reward, done = env.step(action, clf)
            td_error = critic.learn(state, reward, state_)
            actor.learn(state, action, td_error)
            ep_r += reward
            state = state_
            if done:
                print("Summary | Episode: %i , reward: %.3f" % (i, ep_r))
                print("--------------------------------------------------------------------------")
                accHist[i,:] = clf.accHist
                lossHist[i,:] = clf.lossHist
                rewardHist[i] = ep_r
                probs = clf.getAllProb(env.X_valid)

                break
    print("")

    fig = sns.lineplot(x=range(len(rewardHist)), y=rewardHist, label="rewardHist")
    fig.set_title("reward")
    plt.savefig("reward.png")
    plt.show()

    save(accHist[-1][-1])
    save_results(accHist[-1][-1], probs)


def eval():
    restore(0.88, 10, 90)
    # env.render()
    state = env.reset(clf)
    ep_r = 0
    accHist = np.zeros(env.budgets + 1)
    while True:
        env.render()
        action = actor.choose_action(state)
        state_, reward, done = env.step(action, clf)
        ep_r += reward
        state = state_
        if done:
            accHist = clf.accHist
            print("Summary | reward: %.3f" % ep_r)
            print("--------------------------------------------------------------------------")
            sns.lineplot(x=range(len(accHist)), y=accHist, label="accHist")
            plt.show()
            break


if ON_TRAIN:
    train()
else:
    eval()



