# Ape-X

An Implementation of [Distributed Prioritized Experience Replay](https://arxiv.org/abs/1803.00933) (Horgan et al. 2018) in PyTorch.

<img src="https://cl.ly/40b459838c5e/Image%2525202019-03-10%252520at%2525206.53.24%252520PM.png" width="500">

The paper proposes a distributed architecture for deep reinforcement learning with distributed prioritized experience replay. This enables a fast and broad exploration with many actors, which prevents model from learning suboptimal policy.

There are a few implementations which are optimized for powerful single machine with a lot of cores but I tried to implement Ape-X in a multi-node situation with AWS EC2 instances. [ZeroMQ](http://zeromq.org/), [AsyncIO](https://docs.python.org/3/library/asyncio.html), [Multiprocessing](https://docs.python.org/3/library/multiprocessing.html) are really helpful tools for this. 

There are still performance issues with replay server which are caused by the shared memory lock and hyperparameter tuning but this works anyway. Also, there are still some parts  I hard-coded for convenience. I'm trying to improve many parts and really appreciate your help.

# Requirements

```
python 3.7
numpy==1.18.1
torch==1.4.0
pyzmq==19.0.0
opencv-python==4.2.0.32
tensorboard==2.1.0
gym==0.17.0
gym[atari]
```


# Overall Structure

![image](https://user-images.githubusercontent.com/20944657/54428494-069f1700-4761-11e9-96bc-51ba0b8c39e5.png)

# How To Use

## Single Machine

See `run.sh` for details. This forked repo mainly aims as single node training.

## Multi-Node with AWS EC2

not tested

### Thanks to

1. https://github.com/haje01/distper
2. https://github.com/openai/baselines/
