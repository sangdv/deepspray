from PIL import Image 
import glob, os, cv2
import numpy as np
import random, glob
from scipy import ndimage
        
if not os.path.exists("./train"): os.makedirs("./train")
if not os.path.exists("./train/images"): os.makedirs("./train/images")
if not os.path.exists("./train/labels"): os.makedirs("./train/labels")
if not os.path.exists("./valid"): os.makedirs("./valid")
if not os.path.exists("./valid/images"): os.makedirs("./valid/images")
if not os.path.exists("./valid/labels"): os.makedirs("./valid/labels")
if not os.path.exists("./test"): os.makedirs("./test")
if not os.path.exists("./test/images"): os.makedirs("./test/images")

PREFIX = "f_"
# Place original labeled img
SOURCE = "./Scaled-Yolov4/ori_img_2/"
NUM_OBJECT_MIN = 600
NUM_OBJECT_MAX = 1000

noise1 = cv2.imread("./Scaled-Yolov4/noise1.png")
noise1 = cv2.cvtColor(noise1, cv2.COLOR_BGR2RGB)
noise2 = cv2.imread("./Scaled-Yolov4/noise2.png")
noise2 = cv2.cvtColor(noise2, cv2.COLOR_BGR2RGB)
noise3 = cv2.imread("./Scaled-Yolov4/noise3.png")
noise3 = cv2.cvtColor(noise3, cv2.COLOR_BGR2RGB)
ran_noise = random.randint(20, 50)

### GENERATE TRAINING DATA ###
for f in glob.glob('./train/labels.cache'): os.remove(f)
for f in glob.glob('./train/images/*'): os.remove(f)
for f in glob.glob('./train/labels/*'): os.remove(f)

count_img = 0

# Create label pool
img_pool = []
label_pool = []
for img_file in os.listdir(SOURCE + "images"):
    if(img_file[:2]!= "f_"): continue
    img = np.array(Image.open(SOURCE + "images/" + img_file))
    H, W, C = img.shape
    if C==4: img = img[:,:,:3]
    label_txt = open(SOURCE + "labels/"+ img_file[:-4] + ".txt")
    print(SOURCE + "images/" + img_file, img.shape)
    
    for line in label_txt:
        line = line.split()
        label_pool.append(line[0])
        x,y,w,h=float(line[1]),float(line[2]),int(float(line[3])*W),int(float(line[4])*H)
        x1 = int(x*W - w/2)
        y1 = int(y*H - h/2)
        x2 = x1 + w
        y2 = y1 + h
        crop = img[y1:y2, x1:x2, :] # Thêm ảnh object
        img_pool.append(crop) # Thêm kích thước object

        
        
### GENERATE TRANING SET ###
for j in range(1000):
    if(j%100==0): print(j)
    H, W = 1200, 1600
    # Tạo màu nền trắng - xanh đậm - xanh nhạt
    output_img = 255 * np.ones((H, W, 3), np.uint8)
    new_label_txt = open('./train/labels/'+str(j)+'.txt', "w")
    frequence = random.randint(NUM_OBJECT_MIN, NUM_OBJECT_MAX) # Số lượng object mỗi ảnh được gen ra
    for i in range(frequence):
        # Toạ độ mới
        empty = 0
        count_failed = 0
        random_object = 0
        while empty == 0: 
            count_failed += 1
            if(count_failed > 100): break
            random_object = random.randint(0, len(img_pool) - 1)
            crop = img_pool[random_object]
            crop_2 = ndimage.rotate(crop, angle=random.randint(0,180), cval=255)
            x1_, y1_ = random.randint(0, W - crop_2.shape[1]), random.randint(0, H - crop_2.shape[0])
            x2_ = x1_ + crop_2.shape[1]
            y2_ = y1_ + crop_2.shape[0]
            # Nếu là giọt Drop thì cho phép paste luôn không cần xem xét là có đang đè không
            empty = 1
        if(empty == 1):
            output_img[y1_:y2_, x1_:x2_, :][crop_2[:,:,:] < 255] = crop_2[crop_2[:,:,:] < 255]
            new_label_txt.write(label_pool[random_object]+" " + str((x1_+x2_)/2/W) + " " + 
                str((y1_+y2_)/2/H) + " " + str(crop_2.shape[1]/W) + " " + 
                str(crop_2.shape[0]/H) + "\n")
    Image.fromarray(output_img).save('./train/images/'+str(j)+'.png')   
    new_label_txt.close()
    count_img +=1
print("Create Traningset Done!")



### GENERATE VALIDATION SET ###

for f in glob.glob('./valid/labels.cache'): os.remove(f)
for f in glob.glob('./valid/images/*'): os.remove(f)
for f in glob.glob('./valid/labels/*'): os.remove(f)
    
count_img = 0

for j in range(200):
    H, W = 1200, 1600
    # Tạo màu nền trắng - xanh đậm - xanh nhạt
    output_img = 255 * np.ones((H, W, 3), np.uint8)
    new_label_txt = open('./valid/labels/'+str(j)+'.txt', "w")
    frequence = random.randint(NUM_OBJECT_MIN, NUM_OBJECT_MAX) # Số lượng object mỗi ảnh được gen ra
    for i in range(frequence):
        # Toạ độ mới
        empty = 0
        count_failed = 0
        random_object = 0
        while empty == 0: 
            count_failed += 1
            if(count_failed > 100): break
            random_object = random.randint(0, len(img_pool) - 1)
            crop = img_pool[random_object]
            crop_2 = ndimage.rotate(crop, angle=random.randint(0,180), cval=255)
            x1_, y1_ = random.randint(0, W - crop_2.shape[1]), random.randint(0, H - crop_2.shape[0])
            x2_ = x1_ + crop_2.shape[1]
            y2_ = y1_ + crop_2.shape[0]
            # Nếu là giọt Drop thì cho phép paste luôn không cần xem xét là có đang đè không
            empty = 1
        if(empty == 1):
            output_img[y1_:y2_, x1_:x2_, :][crop_2[:,:,:] < 255] = crop_2[crop_2[:,:,:] < 255]
            new_label_txt.write(label_pool[random_object]+" " + str((x1_+x2_)/2/W) + " " + 
                str((y1_+y2_)/2/H) + " " + str(crop_2.shape[1]/W) + " " + 
                str(crop_2.shape[0]/H) + "\n")
    Image.fromarray(output_img).save('./valid/images/'+str(j)+'.png')   
    new_label_txt.close()
    count_img +=1
print("Create Validation Done!")