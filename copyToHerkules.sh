#!/bin/bash

pathRemote="/home/itodaiber/R - Repositories/tmp/"
zipFileName="herkules.zip"

zip -r -q ./$zipFileName config/ lib/ main.py 

scp ."/$zipFileName" herkules:"'$pathRemote'"

ssh herkules "unzip -q '$pathRemote/$zipFileName' -d '$pathRemote'; rm '$pathRemote/$zipFileName'"

rm ./$zipFileName