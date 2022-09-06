#!/bin/bash

cd live_sim
postProcess -func sampleDict -latestTime
cd ..
python3 field2image.py
