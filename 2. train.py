cd Scaled-Yolov4
python train.py --img 1024 --batch 8 --epochs 5000 --data '../data.yaml' --cfg ./models/yolov4-csp.yaml --weights '' --name yolov4-csp-results --cache