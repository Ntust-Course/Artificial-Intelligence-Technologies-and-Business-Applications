"""Network"""
import gym
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from helper import ExperienceBuffer, update_target, update_target_graph

GAMMA = 0.99  # Discount factor.
NUM_EPISODES = 20000  # Total number of episodes to train network for.
TAU = 0.001  # Amount to update target network at each step.
BATCH_SIZE = 32  # Size of training batch
START_EPSILON = 1  # Starting chance of random action
END_EPSILON = 0.1  # Final chance of random action
ANNELING_STEPS = (
    200000
)  # How many steps of training to reduce START_EPSILON to END_EPSILON.
PRE_TRAIN_STEPS = 50000  # Number of steps used before training updates begin.


class QNetwork:
    def __init__(self):
        # These lines establish the feed-forward part of the network used to choose actions
        self.inputs = tf.placeholder(shape=[None, 4], dtype=tf.float32, name="inputs")
        self.temp = tf.placeholder(shape=None, dtype=tf.float32, name="temperature")
        self.keep_percent = tf.placeholder(
            shape=None, dtype=tf.float32, name="keep_percent"
        )

        hidden = slim.fully_connected(
            self.inputs, 64, activation_fn=tf.nn.tanh, biases_initializer=None
        )
        hidden = slim.dropout(hidden, self.keep_percent)
        self.q_out = slim.fully_connected(
            hidden, 2, activation_fn=None, biases_initializer=None
        )

        self.predict = tf.argmax(self.q_out, 1)
        self.q_dist = tf.nn.softmax(self.q_out / self.temp)

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, 2, dtype=tf.float32)

        self.q = tf.reduce_sum(
            tf.multiply(self.q_out, self.actions_onehot), reduction_indices=1
        )

        self.next_q = tf.placeholder(shape=[None], dtype=tf.float32)
        loss = tf.reduce_sum(tf.square(self.next_q - self.q))
        trainer = tf.train.GradientDescentOptimizer(learning_rate=0.0005)
        self.update_model = trainer.minimize(loss)


def choice(exploration, output_graph=False):
    env = gym.make("CartPole-v0")

    tf.reset_default_graph()

    q_net = QNetwork()
    target_net = QNetwork()

    init = tf.global_variables_initializer()
    trainables = tf.trainable_variables()
    target_ops = update_target_graph(trainables, TAU)
    my_buffer = ExperienceBuffer()

    # create lists to contain total rewards and steps per episode
    step_list = step_means = reward_list = reward_means = []
    with tf.Session() as sess:
        if output_graph:
            # $ tensorboard --logdir=logs
            # http://0.0.0.0:6006/
            tf.summary.FileWriter("logs/", sess.graph)
        sess.run(init)
        update_target(target_ops, sess)
        epsilon = START_EPSILON
        step_drop = (START_EPSILON - END_EPSILON) / ANNELING_STEPS
        total_steps = 0

        for episode in range(NUM_EPISODES):
            observation = env.reset()
            rewards = 0
            done = False
            for j in range(1000):
                if exploration == "greedy":
                    # Choose an action with the maximum expected value.
                    actions, all_q = sess.run(
                        [q_net.predict, q_net.q_out],
                        feed_dict={
                            q_net.inputs: [observation],
                            q_net.keep_percent: 1.0,
                        },
                    )
                    action = actions[0]
                if exploration == "random":
                    # Choose an action randomly.
                    action = env.action_space.sample()
                if exploration == "e-greedy":
                    # Choose an action by greedily (with e chance of random action) from the Q-network
                    if np.random.rand(1) < epsilon or total_steps < PRE_TRAIN_STEPS:
                        action = env.action_space.sample()
                    else:
                        action, all_q = sess.run(
                            [q_net.predict, q_net.q_out],
                            feed_dict={
                                q_net.inputs: [observation],
                                q_net.keep_percent: 1.0,
                            },
                        )
                        action = action[0]
                if exploration == "boltzmann":
                    # Choose an action probabilistically, with weights relative to the Q-values.
                    q_d, all_q = sess.run(
                        [q_net.q_dist, q_net.q_out],
                        feed_dict={
                            q_net.inputs: [observation],
                            q_net.temp: epsilon,
                            q_net.keep_percent: 1.0,
                        },
                    )
                    action = np.random.choice(q_d[0], p=q_d[0])
                    action = np.argmax(q_d[0] == action)
                if exploration == "bayesian":
                    # Choose an action using a sample from a dropout approximation of a bayesian q-network.
                    actions, all_q = sess.run(
                        [q_net.predict, q_net.q_out],
                        feed_dict={
                            q_net.inputs: [observation],
                            q_net.keep_percent: (1 - epsilon) + 0.1,
                        },
                    )
                    action = actions[0]
                if exploration == "bayesian-improvised":
                    # Choose an action using a sample from a dropout approximation of a bayesian q-network.
                    actions, all_q = sess.run(
                        [q_net.predict, q_net.q_out],
                        feed_dict={
                            q_net.inputs: [observation],
                            q_net.keep_percent: (1 - np.sqrt(epsilon)) + 0.01,
                        },
                    )
                    action = actions[0]

                # Get new state and reward from environment
                env.render()
                observation_next, reward, done, _ = env.step(action)
                my_buffer.add(
                    np.reshape(
                        np.array([observation, action, reward, observation_next, done]),
                        [1, 5],
                    )
                )

                if epsilon > END_EPSILON and total_steps > PRE_TRAIN_STEPS:
                    epsilon -= step_drop

                if total_steps > PRE_TRAIN_STEPS and total_steps % 5 == 0:
                    # We use Double-DQN training algorithm
                    train_batch = my_buffer.sample(BATCH_SIZE)
                    q1 = sess.run(
                        q_net.predict,
                        feed_dict={
                            q_net.inputs: np.vstack(train_batch[:, 3]),
                            q_net.keep_percent: 1.0,
                        },
                    )
                    q2 = sess.run(
                        target_net.q_out,
                        feed_dict={
                            target_net.inputs: np.vstack(train_batch[:, 3]),
                            target_net.keep_percent: 1.0,
                        },
                    )
                    end_multiplier = -(train_batch[:, 4] - 1)
                    double_q = q2[range(BATCH_SIZE), q1]
                    target_q = train_batch[:, 2] + (GAMMA * double_q * end_multiplier)
                    _ = sess.run(
                        q_net.update_model,
                        feed_dict={
                            q_net.inputs: np.vstack(train_batch[:, 0]),
                            q_net.next_q: target_q,
                            q_net.keep_percent: 1.0,
                            q_net.actions: train_batch[:, 1],
                        },
                    )
                    update_target(target_ops, sess)

                rewards += reward
                observation = observation_next
                total_steps += 1
                if done:
                    break
            step_list.append(j)
            reward_list.append(rewards)
            if episode % 100 == 0 and episode != 0:
                reward_mean = np.mean(reward_list[-100:])
                step_mean = np.mean(step_list[-100:])
                print_str = f"Mean Reward: {reward_mean} Total Steps: {total_steps}"
                if exploration == "e-greedy":
                    print_str += f" e: {epsilon}"
                if exploration == "boltzmann":
                    print_str += f" t: {epsilon}"
                if "bayesian" in exploration:
                    print_str += f" p: {epsilon}"
                print(print_str)
                reward_means.append(reward_mean)
                step_means.append(step_mean)
    print(
        f"Percent of succesful episodes: {sum(reward_list) / NUM_EPISODES}"
        f"% Exploration Approach {exploration}"
    )
    return reward_means, step_means
