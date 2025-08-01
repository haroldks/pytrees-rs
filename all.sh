#!/usr/bin/env bash

TIME_LIMIT=(15 30 60 120 180 240 300)

for timeout in "${TIME_LIMIT[@]}"; do
    ./expegains.sh --timeout $timeout
done
