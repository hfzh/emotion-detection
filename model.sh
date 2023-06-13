FILE=$HOME/emotion-detection/emotion-detection-service/app/model/enet_b2_8.onnx

mkdir -p $HOME/emotion-detection/emotion-detection-service/app/model

if [ -f "$FILE" ]; then
    echo "Models exist."
else
    wget -O $FILE \
    https://github.com/HSE-asavchenko/face-emotion-recognition/raw/main/models/affectnet_emotions/onnx/enet_b2_8.onnx 
fi
