docker run --rm -it \
    --shm-size=8g \
    --gpus all \
    -v $(pwd)/src:/app \
    -v $(pwd)/data:/data \
    --name train-seg-chad \
    segmentation /app/train.py