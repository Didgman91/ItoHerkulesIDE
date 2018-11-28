#!/bin/bash

pathRemote="/home/itodaiber/R - Repositories/tmp/"

./copyToHerkules.sh

ssh herkules "cd '$pathRemote'; bsub 'python3.4 main.py'"