import gymnasium as gym
import ale_py
import matplotlib.pyplot as plt
import numpy as np
import pickle
from IPython.display import clear_output
import time


def display_frame(frame):
    """Display a single frame using matplotlib"""
    plt.figure(figsize=(8, 6))
    plt.imshow(frame)
    plt.axis('off')
    plt.show()


gym.register_envs(ale_py)


# Copied / inspired from the classic https://karpathy.github.io/2016/05/31/rl/:
#  MIT License
#   https://gist.githubusercontent.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5/raw/06d092624118444f7350a22653e8ba9d1c6e63d6/pg-pong.py

#  Hidden layer neurons
H = 200

# Batch size is the number of episodes we use to param update
batch_size = 10

# Learning rate is the step size 
learning_rate = 0.0001

# Gamma is the classic discount factor for reward used in the Bellman equation
gamma = 0.99

# ??
decay_rate = 0.99

# Checkpoint resume / load model
resume = False

# Render the game on screen - of course, if enabled during training, caps the speed of training!
render = False

# Initialize model

D = 80 * 80

if resume:
  model = pickle.load(open('save.p', 'rb'))
else:
  model = {}
  # This is implementing a variant of "Xavier" (also called Glorot) initialization, where:
  # Weights are randomly sampled from a normal distribution
  # Then scaled by dividing by the square root of the input dimension size
  model['W1'] = np.random.randn(H, D) / np.sqrt(D)
  model['W2'] = np.random.randn(H) / np.sqrt(H)
# Now model has two arrays W1 and W2
#  W1 200x6400 tracks the weights from input (6400 pixels) to the hidden layer (200 neurons), 
#  W2 200x1 from hidden to output (convolved to a single output (action) value UP or DOWN)

# print("Model W1: " + str(model['W1']))
# print("Model W2: " + str(model['W2']))
grad_buffer = { k: np.zeros_like(v) for k,v in model.items() }
# print("Grad buffer: " + str(grad_buffer))
rmsprop_cache = { k: np.zeros_like(v) for k,v in model.items() }
# print("RMS prop cache: " + str(rmsprop_cache))

def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x))

def preprocess(I):
  # Crop, scale down, remove bg 1/2, only set 1 for paddles and ball. This includes the score too!
  I = I[35:195]
  I = I[::2,::2,0]
  I[I == 144] = 0
  I[I == 109] = 0
  I[I != 0] = 1
  return I.astype(np.float32).ravel()

def discount_rewards(r):
  # We are doing the classic Bellman equation here: (gamma^t * R_t + gamma^(t+1) * R_t+1 + gamma^(t+2) * R_t+2 + ...)
  discounted_r = np.zeros_like(r)
  running_add = 0
  # Reverse so we favor weighting recent rewards more
  for t in reversed(range(0, r.size)):
     # Pong specific reset
     if (r[t] != 0): running_add = 0
     running_add = running_add * gamma + r[t]
     discounted_r[t] = running_add
  return discounted_r

def policy_forward(x):
  # h = Sum(W1_i * x_i)
  h = np.dot(model['W1'], x)
  # ReLU non-linearity which clips values below 0 to 0.
  h[h<0] = 0
  # Second layer:  logp = W2 . h
  logp = np.dot(model['W2'], h)
  # Note that this produces a singular scalar value which we interpret as the log-odds of taking action 2 (UP) vs action 3 (DOWN)
  # We then apply the sigmoid function to produce a probability between 0 and 1
  p = sigmoid(logp)
  return p, h

# Back-propagate: From the episode_logp -> episode_hidden -> episode_input
def policy_backward(episode_hidden, episode_logp, episode_input):
  # dW2 = (Transpose of hidden layer) . (gradient of log-probabilities)
  dW2 = np.dot(episode_hidden.T, episode_logp).ravel()
  # A matrix outer product of the input to the hidden layer and the gradient of the log-probabilities
  dHidden = np.outer(episode_logp, model['W2'])
  # ReLU backprop
  dHidden[episode_hidden <= 0] = 0
  # Finally, the gradient for W1 is the matrix multiplication of the transposed input vector
  dW1 = np.dot(dHidden.T, episode_input)
  return {'W1': dW1, 'W2': dW2}

# Following is mostly verbatim from Karpathy's code
env = gym.make("ALE/Pong-v5", render_mode="rgb_array" if render else None)
observation, info = env.reset()
prev_x = None
x_list,h_list,dlogp_list,dreward_list = [],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 0
log_count = 0
delay = 0.033 # 30fps
num_steps = 0

Run = True
while Run:
  # Step 1: Render the current frame as is (including if this is the first frame which sets up the game)


  # Step 2: Preprocess the observation, set input to network to be difference image
  # Convert to grayscale/downsample
  cur_x = preprocess(observation)

  if render:
      frame = env.render()
      # Clear previous output and show the new frame
      clear_output(wait=True)
      # display_frame(frame)      
      preproc_frame = cur_x.reshape(80, 80)

      for row in preproc_frame:
        line = ''
        for cell in row:
          line += '█' if cell != 0 else ' '
        print(line)
      display_frame(preproc_frame)      
      # Add a small delay to make visualization visible
      time.sleep(delay)  
  # X = (Cur - Prev) (or 0 for the first frame)
  x = cur_x - prev_x if prev_x is not None else np.zeros(D)
  # Diff the input so the NN sees chanes (i.e. where the ball was/is, paddles were/are so it can compute the 
  # velocity and direction of movement and hence the final actions)
  prev_x = cur_x
  
  # Step 3: Run the policy to obtain an action. If it's untrained, we are sampling a random action.
  # Forward the policy network; sample an action from the returned probability
  aprob, h = policy_forward(x)
  # Logit probability of taking action 2 (UP)
  action = 2 if np.random.uniform() < aprob else 3
  x_list.append(x)
  h_list.append(h)
  y = 1 if action == 2 else 0
  dlogp_list.append(y - aprob)

  # Step 4: Step through the environment with that action and
  #         obtain the new observation, reward and a signal for if the game is over (i.e. done: won/lost)
  observation, reward, terminated, truncated, info = env.step(action)
  reward_sum += reward
  dreward_list.append(reward)
  
  # Step 5: Once an episode is done (i.e. game over), we perform the following:
  #         - compute the discounted reward backwards through time
  #         - compute the policy gradient which is really the 
  if terminated or truncated:
    # Run = False
    episode_number += 1
    num_steps = len(dreward_list)
    # Add the episode to the batch. Each batch has N episodes; Each episode has M steps; Each step has 6400 observations, 2x200 hidden states, 1 gradient; 
    episode_x = np.vstack(x_list)
    episode_hidden = np.vstack(h_list)
    episode_logp = np.vstack(dlogp_list)
    episode_reward = np.vstack(dreward_list)
    # Episode is added to the batch, now reset for the next episode.
    x_list,h_list,dlogp_list,dreward_list = [],[],[],[]

    # Compute the discounted rewards i.e. R_t = sum(gamma^k * r_t+k)
    discounted_epr = discount_rewards(episode_reward)
    # Standardize the rewards to be unit normal (helps control the gradient estimator variance)
    discounted_epr -= np.mean(discounted_epr)
    discounted_epr /= np.std(discounted_epr) 

    # This is the policy gradient magic: If we won, then we 'train' the policy (i.e. states->action probability map)
    # to inch towards the action we took. Same goes for if we lost (in the opposite direction)
    episode_logp *= discounted_epr
    gradient = policy_backward(episode_hidden, episode_logp, episode_x)

    # Now we accumulate the gradients over the batch. 
    # For the model's {W1, W2}, we are summing the gradients.
    for k_weight in model:
      grad_buffer[k_weight] += gradient[k_weight]

    # When we have enough episodes in a batch, perform a param update using RMSProp (https://cs231n.github.io/neural-networks-3/#ada)
    if episode_number % batch_size == 0:
      for k_weight,val in model.items():
         dx = grad_buffer[k_weight]
         rmsprop_cache[k_weight] = (decay_rate * rmsprop_cache[k_weight]) + (1 - decay_rate) * (dx**2)
         model[k_weight] += (learning_rate * dx) / (np.sqrt(rmsprop_cache[k_weight]) + 1e-5)
         grad_buffer[k_weight] = np.zeros_like(val)
    # Output internal state such as reward etc
    if running_reward is None:
      running_reward = reward_sum
    else:
      # Exponential Moving Average (EMA) of reward; 99% previous, 1% current.
      running_reward = (running_reward * 0.99) + (reward_sum * 0.01)
    print(f'Episode Reset {episode_number} # {num_steps} steps: reward total was {reward_sum}. Running mean: {running_reward}')
    if episode_number % 100 == 0: 
       pickle.dump(model, open('save.p', 'wb'))

    reward_sum = 0
    observation, info = env.reset()
    prev_x = None
  if reward != 0:
    if log_count % 30 == 0:
      log_count += 1
      print(f'Episode {episode_number}: game finished, reward: {reward}.')
  



