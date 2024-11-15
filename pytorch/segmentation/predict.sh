docker run --rm -it \
    --gpus all \
    -v $(pwd)/src:/app \
    -v $(pwd)/data:/data \
    segmentation /app/predict.py