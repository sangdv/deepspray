# Detect from best checkpoint
cd Scaled-Yolov4
python detect.py --weights ./runs/exp0_yolov4-csp-results/weights/best.pt --img 1024 --conf 0.1 --source ../test/images

# Detect from last checkpoint when trainning
# cd Scaled-Yolov4
# cp ./runs/exp31_yolov4-csp-results/weights/last.pt ./runs/exp31_yolov4-csp-results/weights/last2.pt 
# python detect.py --weights ./runs/exp0_yolov4-csp-results/weights/last2.pt --img 1920 --conf 0.1 --source ../test/images