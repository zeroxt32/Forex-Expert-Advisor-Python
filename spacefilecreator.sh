#!/bin/bash

# Set the directory where the files will be created
dir="./test"

# Loop to create the files
for i in {1..300}; do
    filename="file $i.txt"
    touch "$dir/$filename"
done

