# Udacity-DRL-Project-3

PPO Algorithm to solve Udacity DRL Project 3 Tennis Environment.

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The environment is considered solved once the average score across 100 episodes is >= 0.5.

Getting Started
Download the environment via one of the links below (select the correct operating system).

Linux: (https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
Mac OSX: (https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
Windows (32-bit): (https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
Windows (64-bit): (https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

Place the file in the same location of the repository files and unzip the contents (this should create a folder/path consistent with the path contained within Report.ipynb).

Running the Code
The code is run entirely within the Jupyter Notebook Report.ipynb, this file should be followed sequentially.

Dependencies
The following is required to run this repository correctly:

Python 3.6.1 or higher
Numpy
Matplotlib
Unity ML-Agents Toolkit (https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Installation.md)
