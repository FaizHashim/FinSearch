import tensorflow as tf
import numpy as np
import gym
from tensorflow import keras
from collections import deque

import random

# env = gym.make('Pendulum-v1')
env = gym.make('Pendulum-v1', render_mode="human")

input_shape = env.observation_space.shape[0]
# num_actions = env.action_space.n
print(env.action_space)
# print(env.observation_space.shape[0])
num_actions = 9


def getContAction(discAction):
  return ((discAction / (num_actions-1)) * 4)-2

value_network = tf.keras.models.Sequential([
    tf.keras.layers.Reshape((3,1), input_shape=(3,)),
    tf.keras.layers.Conv1D(filters=8, kernel_size=2, activation='relu'),
    tf.keras.layers.Conv1D(filters=16, kernel_size=2, activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(num_actions, activation='softmax'),
])


# Set up the optimizer and loss function
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.MeanSquaredError(reduction="sum_over_batch_size", name="mean_squared_error")
#value_network = tf.keras.models.load_model('keras')

num_episodes = 500
epsilon = 1
gamma = 0.95
state = env.reset()
batch = 200
replay = deque(maxlen=2000)
epoch = 0
alpha = 0.1

# print(state[0])
# print(np.array(state[0]).shape)
# test = value_network.predict(np.array(state[0]).reshape(1, 3))
# value_function = value_network.predict(np.array(state[0]),verbose=0)[0]
# print(test)

for episode in range(num_episodes):

    state = env.reset()
    state = state[0]
    # Run the episode
    print(f"Episode : {episode}")
    while True:
        value_function = value_network.predict(np.array(state).reshape(1,3),verbose=0)[0]

        if np.random.rand()>epsilon:
            action = np.argmax(value_function)

        else:
            action = np.random.choice(num_actions)

        cont_action = np.array([getContAction(action)])

        next_state, reward, done1, done2, _ = env.step(cont_action)
        done = 1 if done1 else 0
        replay.append((state,action,reward,next_state,done))
        state = next_state.copy()


        if done:
            break

        if len(replay)>batch:
            with tf.GradientTape() as tape:
                batch_ = random.sample(replay,batch)
                q_value1 = value_network(tf.convert_to_tensor([x[0] for x in batch_]))
                q_value2 = value_network(tf.convert_to_tensor([x[3] for x in batch_]))

                reward = tf.convert_to_tensor([x[2] for x in batch_])
                action = tf.convert_to_tensor([x[1] for x in batch_])
                done =   tf.convert_to_tensor([x[4] for x in batch_])

                actual_q_value1 = tf.cast(reward,tf.float64) + tf.cast(tf.constant(alpha),tf.float64)*(tf.cast(tf.constant(gamma),tf.float64)*tf.cast((tf.constant(1)-done),tf.float64)*tf.cast(tf.reduce_max(q_value2),tf.float64))
                loss = tf.cast(tf.gather(q_value1,action,axis=1,batch_dims=1),tf.float64)
                loss = loss - actual_q_value1
                loss = tf.reduce_mean(tf.math.pow(loss,2))


                grads = tape.gradient(loss, value_network.trainable_variables)
                optimizer.apply_gradients(zip(grads, value_network.trainable_variables))

                print('Episode {} Epoch {} done with loss {} !!!!!!'.format(episode, epoch,loss))
                value_network.save('dqn_cnn.keras')
                if epoch%100==0:
                    epsilon*=0.999
                epoch+=1

                if epoch % 500 == 0:
                    break
