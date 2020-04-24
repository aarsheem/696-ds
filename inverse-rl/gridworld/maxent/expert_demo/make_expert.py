import sys
sys.path.append("../")
import numpy as np
from gridworld import Gridworld


#Actions: up (0), down (1), left (2), right (3), do nothing (4)

human_policy = [3,3,3,3,1,\
                0,3,3,3,1,\
                0,1,3,3,1,\
                0,2,0,3,1,\
                0,2,3,3,0]


expert = []
for _ in range(10000):
    env = Gridworld()
    ep = []
    i = 0
    s = 0
    while i<50:
        a = human_policy[s]
        s,reward,done,_ = env.step(a)
        ep.append([s, a, reward])
        if done:
            break
        i+=1
    # print(i)
    ep = np.array(ep)
    expert.append(ep)
print(expert)
expert_np = np.array(expert)
np.save("expert_demo", expert_np)
print(expert_np.shape)