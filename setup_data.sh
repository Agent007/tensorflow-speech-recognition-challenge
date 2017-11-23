#!/bin/bash -e

cd data/download
kg download
7z x -o../ train.7z
7z x -o../ test.7z

