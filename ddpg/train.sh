#!/bin/bash

python3 ddpg.py --minibatch-size=64 --buffer-size=1000000 --env=Pendulum-v0 --render-env 

