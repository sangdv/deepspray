# YOLOv4-large

## Directory structure
- Scaled-Yolov4/ori_img: original annotated data
- train, valid: synthesized training và validation images
- test: test data

## Step 1: Installation lib

```bash
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install -U PyYAML
pip install opencv-python
sudo apt-get install python-scipy
pip install Pillow scipy matplotlib
cd mish-cuda
python setup.py build install
cd ..
```

## Step 2: Generate Train/Valid Data

```bash

python generate_data.py

```

## Step 3. Training

```bash
cd Scaled-Yolov4
python train.py --img 1024 --batch 8 --epochs 5000 --data '../data.yaml' --cfg ./models/yolov4-csp.yaml --weights '' --name yolov4-csp-results --cache
```

## Step 4. Inference

To run best checkpoint:
```bash
cd Scaled-Yolov4
python detect.py --weights ./runs/exp0_yolov4-csp-results/weights/best.pt --img 1024 --conf 0.1 --source ../test/images
```

To run last checkpoint:
```bash
cd Scaled-Yolov4
cp ./runs/exp31_yolov4-csp-results/weights/last.pt ./runs/exp31_yolov4-csp-results/weights/last2.pt 
python detect.py --weights ./runs/exp0_yolov4-csp-results/weights/last2.pt --img 1920 --conf 0.1 --source ../test/images
```

## All-in-one file

You also can use all-in-one file in "train_scaled_yolov4_full_step.ipynb"
