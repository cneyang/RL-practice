#!/bin/bash
timestamp() {
    date +"%T"
}

timestamp
python3 pacman.py -x 10000

timestamp
python3 pacman.py -x 10000 --double

timestamp
python3 pacman.py -x 10000 --double --multistep 3

timestamp
