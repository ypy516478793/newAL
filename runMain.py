from environment import DataEnv
from AC import Actor, Critic
from CNN_classifier import Classifier
import tensorflow as tf

MAX_EPISODES = 900
ON_TRAIN = True
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
    saver.save(sess, './params/AC', write_meta_graph=False)


def restore():
    saver = tf.train.Saver()
    saver.restore(sess, './params/AC')

def train():
    # start training
    for i in range(MAX_EPISODES):
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
                print('Ep: %i | %s | ep_r: %.1f' % (i, '---' if not done else 'done', ep_r))
                break
    save()


def eval():
    restore()
    # env.render()
    state = env.reset(clf)
    while True:
        env.render()
        action = actor.choose_action(state)
        state, reward, done = env.step(action, clf)


if ON_TRAIN:
    train()
else:
    eval()



