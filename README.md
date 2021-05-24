# YOLOv4-large

## Directory structure
- train, test, valid: Directory containing model traning data - includes photos and corresponding labels
- Scaled-Yolov4/ori_img: Folder containing the tagged original image data

## Step 1: Install

```bash
conda create -n deepspray python=3.6
conda activate deepspray
```

```bash
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install tensorflow==1.15.0
pip install -U PyYAML
pip install opencv-python
sudo apt-get install python-scipy
pip install Pillow scipy matplotlib
cd mish-cuda
python setup.py build install
cd ..
```

## Step 2: Generate Train/Valid Data from source in Scaled-Yolov4/ori_img

```bash
python gen_data_ver_2.py 
```

## Step 3. Training

```bash
cd Scaled-Yolov4
python train.py --img 1200 --batch 16 --epochs 5000 --data '../data.yaml' --cfg ./models/yolov4-csp.yaml --weights '' --name yolov4-csp-results --cache
```

## Step 4. Testing

To run best checkpoint:
```bash
cd Scaled-Yolov4
python detect.py --weights ./runs/exp1_yolov4-csp-results/weights/best.pt --img 1200 --conf 0.1 --source ../test/images
```

To run last checkpoint: (assuming exp1_yolov4 is the last running)
```bash
cd Scaled-Yolov4
cp ./runs/exp1_yolov4-csp-results/weights/last.pt ./runs/exp1_yolov4-csp-results/weights/last2.pt 
python detect.py --weights ./runs/exp1_yolov4-csp-results/weights/last2.pt --img 1200 --conf 0.1 --source ../test/images
```

## Citation

```
@article{wang2020scaled,
  title={{Scaled-YOLOv4}: Scaling Cross Stage Partial Network},
  author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
  journal={arXiv preprint arXiv:2011.08036},
  year={2020}
}
```
