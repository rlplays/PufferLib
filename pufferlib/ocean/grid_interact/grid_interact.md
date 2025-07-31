# Grid Interact

- A 2D WxH Grid consisting of
 - `#` -> Wall
 - `0` -> Agent 0
 - `1` -> Agent 1
 - `R` -> Reward
 - `G` -> Goal
 - ` ` -> Empty space.

Goal is for the Agent X to avoid Agent Y, collect all rewards and reach the goal.
If the agent reaches the goal first before collecting the reward, the game is over (both agents lose).
If Agent X touches Agent Y that agent X gets a small negative reward.


Agent Action Space: Discrete 0, 1, 2, 3 (UP, DOWN, LEFT, RIGHT)
Each Agent's observation: an WxH (1-dim shaped) ego-centric observation 1-hot encoded 

```js
###########################
#  ######        R        #
#      #####        ####  #
#                  0      #
#   R    #    ######      #
#        # G              #
#        #     #      R   #
#     ######        1     #
#                         #
###########################
```

Whichever agent X or Y (0 or 1 here- could be more than two as well) reaches the goal first after collecting the rewards gets the final reward multiplier of 10.
The other agent gets a reward multiplier of 0.

 


Things to add later:
- Hitting the wall (when you know it's in front of you) gets you a -0.5 reward
- Going to the same place over and over again earns a reward of -0.5 (collect last-known 5 locations per agent and check)
- Bonus: Add a W weapon - an agent collecting W can touch and 'kill' the other agent.

# Quick commands to copy/pasta

Build/run program alone:

```bash
bash scripts/build_ocean.sh grid_interact && ./grid_interact
```


Train/export:

```bash
python -m pufferlib.pufferl train puffer_grid_interact --train.device cuda 
```
