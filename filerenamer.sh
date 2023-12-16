#!/bin/bash

# Set the maximum number of files to rename at once
batch_size=100

# Get a list of files to rename
files=(./M1M5Charts/*)

# Loop over the files in batches and rename them
for (( i=0; i<${#files[@]}; i+=batch_size )); do
    batch=("${files[@]:i:batch_size}")
    rename 's/ /_/g' "${batch[@]}"
done
