FILE=$HOME/emotion-detection/emotion-detection-service/app/model/enet_b2_8_best.pt

mkdir -p $HOME/emotion-detection/emotion-detection-service/app/model

if [ -f "$FILE" ]; then
    echo "Models exist."
else
    wget -O $FILE \
    https://github.com/HSE-asavchenko/face-emotion-recognition/raw/main/models/affectnet_emotions/enet_b2_8_best.pt
fi
