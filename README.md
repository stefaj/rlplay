# Notes
Personal repository for playing around with reinforcement learning

## Deep Q

### Tricks
Overall focus seems to be to make problem more similar to supervised learning,
by keeping the data more stationary.

Use separate target network for calculating the target. 
Copy weights over from main network to target network occasionally.
NNs alias between similar states. 
According to Mnih, larger NNs are less prone to chasing their own tail, as they alias less.

Use replay memory. Store transitions in replay memory, and batch update on that.


## Policy Gradient
