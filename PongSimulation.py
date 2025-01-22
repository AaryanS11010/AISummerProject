import gymnasium as gym
import ale_py  
import pickle  #First Change
import numpy as np #Second Change

# Load the trained model
model = pickle.load(open('save.p', 'rb'))

def sigmoid(x): 
  return 1.0 / (1.0 + np.exp(-x))

def prepro(I):
  I = I[35:195]
  I = I[::2, ::2, 0]
  I[I == 144] = 0
  I[I == 109] = 0
  I[I != 0] = 1
  return I.astype(np.float32).ravel()

def policy_forward(x):
  h = np.dot(model['W1'], x)
  h[h < 0] = 0
  logp = np.dot(model['W2'], h)
  p = sigmoid(logp)
  return p, h

env = gym.make("Pong-v4", render_mode='human') # Set render_mode to 'human' to see the game
observation, info = env.reset()
prev_x = None

while True:
  # preprocess the observation, set input to network to be difference image
  cur_x = prepro(observation)
  x = cur_x - prev_x if prev_x is not None else np.zeros(80*80)
  prev_x = cur_x

  # forward the policy network and sample an action from the returned probability
  aprob, h = policy_forward(x)
  action = 2 if np.random.uniform() < aprob else 3

  # step the environment and get new measurements
  observation, reward, done, truncated, info = env.step(action)
  done = done or truncated

  if done:
    observation, info = env.reset()
    prev_x = None

