#!/bin/bash

ASSETS_PATH=$HOME/emotion-detection/emotion-detection-service/app/assets
MODEL_FILE=$ASSETS_PATH/enet_b2_8_best.pt
HAAR_FILE=$ASSETS_PATH/haarcascade_frontalface_default.xml

mkdir -p $ASSETS_PATH

if [ -f "$MODEL_FILE" ]; then
    echo "Models exist."
else
    wget -O $MODEL_FILE \
    https://github.com/HSE-asavchenko/face-emotion-recognition/raw/main/models/affectnet_emotions/enet_b2_8_best.pt
fi
