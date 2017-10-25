import gym
import numpy as np

env = gym.make('CartPole-v0')
env.reset()
action = env.action_space.sample()
params = np.random.rand(4)*2-1

def run_episode(env, params):
    obs = env.reset()
    total = 0
    for i in range(0,1000):
        ans = np.matmul(params, obs)
        action = 1 if ans > 0 else 0
        obs, reward, done, info = env.step(action)
        total += reward
        # if done: break
    return total

best_params = params
best_total = 0

for i in range(0,100):
    params = np.random.rand(4)*2-1
    env.reset()
    total = run_episode(env, params)
    if total > best_total:
        print("New total %d" % total)
        (best_params, best_total) = (params, total)

obs = env.reset()
for i in range(0,1000):
   env.render()
   ans = np.matmul(best_params, obs)
   action = 1 if ans > 0 else 0
   obs, reward, done, info = env.step(action)

