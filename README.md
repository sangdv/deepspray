# YOLOv4-large

## Directory structure
- train, test, valid: training set, test set and validation set. Each includes images and annotations
- data/data_1: original annotated data

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

NOTE: if you want to train with gray image plz add --gray flag to tep 2,3,4.

## Step 1: Generate background image (please ignore if you have bg image)
(In this sample code I provide 3 types of backgound: white, with noise, and 3 solid color profiles, create for 2 types of objects and save in 2 backgound folders: ./data/backgound_1 and ./data/backgound_2)

```bash
cd data
python noise.py
```

## Step 2: Generate Train/Valid Data from annotated in ./data/ori_img_3

```bash
python generate_dataset.py --train_num 1000 --valid_num 200 --empty --source "./data/data_1" --background "./data/background_1/"
python generate_dataset.py --train_num 1000 --valid_num 200 --thresh_pixel 430 --source "./data/data_2" --background "./data/background_2/" 
```

```
parser.add_argument('--train_num', type=int, default=1000, help='Number of traning images')
parser.add_argument('--valid_num', type=int, default=200, help='Number of validation images')
parser.add_argument('--images', type=str, default="./data/data_1/images", help='path to source images folder')
parser.add_argument('--labels', type=str, default="./data/data_1/labels", help='path to source labels folder')
parser.add_argument('--background', type=str, default="./data/background_1/", help='path to source background folder')
parser.add_argument('--opt.num_object_max', type=int, default=1100, help='Maximum number of objects per images')
parser.add_argument('--opt.num_object_min', type=int, default=900, help='Maximum number of objects per images')
parser.add_argument('--gray', action='store_true', help='Gray or RGB image')
parser.add_argument('--empty', action='store_true', help='Empty train/valid folder or not')
```

## Step 3. Training

Please edit `./data.yaml`:
```
nc: 5 # Number of class
names: ['bag', 'lobe', 'Detached ligament', 'drop', 'Attached ligament'] # Class list name
```

```bash
python train.py --img 704 --batch 8 --epochs 1000 --data './data.yaml' --cfg ./models/yolov4-csp.yaml --weights '' --name yolov4-csp-results --cache
```

### Arg for traning step:
```
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov4-p5.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='', help='hyperparameters path, i.e. data/hyp.scratch.yaml')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='train,test sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const='get_last', default=False,
                        help='resume from given path/last.pt, or most recent run if blank')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--logdir', type=str, default='runs/', help='logging directory')
    opt = parser.parse_args()
```

## Step 4. Inference

To test with the best checkpoint:
```bash
cd Scaled-Yolov4
python detect.py --weights ./runs/exp23_yolov4-csp-results/weights/last.pt --img 720 --conf 0.3 --source ../test/images
```

### Arg for testing step:

```
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov4-p5.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source image to test')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
```

## Check status:

Please run `plot_train_image.ipynb` and `visualize_epoch.ipynb` to check annotation data and show score of training realtime

## Citation

```
@article{wang2020scaled,
  title={{Scaled-YOLOv4}: Scaling Cross Stage Partial Network},
  author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
  journal={arXiv preprint arXiv:2011.08036},
  year={2020}
}
```
