from environment import DataEnv
from AC import Actor, Critic
from CNN_classifier import Classifier
import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

MAX_EPISODES = 100
ON_TRAIN = False
OUTPUT_GRAPH = False
LR_A = 0.001    # learning rate for actor
LR_C = 0.01

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

if OUTPUT_GRAPH:
    tf.summary.FileWriter("logs/", sess.graph)

steps = []


def save():
    saver = tf.train.Saver()
    saver.save(sess, './RLmodel/AC', write_meta_graph=False)


def restore():
    saver = tf.train.Saver()
    saver.restore(sess, './RLmodel/AC')

def train():
    accHist = np.zeros([MAX_EPISODES, env.budgets + 1])
    lossHist = np.zeros([MAX_EPISODES, env.budgets + 1])
    rewardHist = np.zeros([MAX_EPISODES])
    # start training
    for i in range(MAX_EPISODES):
        clf.reset()
        state = env.reset(clf)
        ep_r = 0.
        while True:
            # env.render()

            action = actor.choose_action(state)

            state_, reward, done = env.step(action, clf)

            td_error = critic.learn(state, reward, state_)

            actor.learn(state, action, td_error)

            ep_r += reward

            state = state_
            if done:
                print("Summary | Episode: %i , reward: %.3f" % (i, ep_r))
                print('--------------------------------------------------------------------------')

                accHist[i,:] = clf.accHist
                lossHist[i,:] = clf.lossHist
                rewardHist[i] = ep_r
                break

    print("")

    import pickle
    with open('history.pickle', 'wb') as f:
        pickle.dump([accHist, lossHist, rewardHist], f)

    fig = sns.lineplot(x=range(len(rewardHist)), y=rewardHist, label='rewardHist')
    fig.set_title('reward')
    plt.savefig("reward.png")
    plt.show()

    save()


def eval():
    restore()
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
            print('--------------------------------------------------------------------------')
            sns.lineplot(x=range(len(accHist)), y=accHist, label='accHist')
            plt.show()
            break


if ON_TRAIN:
    train()
else:
    eval()



