#!/bin/bash
tmux new-session -d 'make train'
tmux split-window -v 'make tensorboard'
tmux -2 attach-session -d 
