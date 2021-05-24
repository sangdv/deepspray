from PIL import Image 
import glob, os, cv2, random, glob
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

if not os.path.exists("./train"): os.makedirs("./train")
if not os.path.exists("./train/images"): os.makedirs("./train/images")
if not os.path.exists("./train/labels"): os.makedirs("./train/labels")
if not os.path.exists("./valid"): os.makedirs("./valid")
if not os.path.exists("./valid/images"): os.makedirs("./valid/images")
if not os.path.exists("./valid/labels"): os.makedirs("./valid/labels")
if not os.path.exists("./test"): os.makedirs("./test")
if not os.path.exists("./test/images"): os.makedirs("./test/images")

for f in glob.glob('./train/labels.cache'): os.remove(f)
for f in glob.glob('./train/images/*'): os.remove(f)
for f in glob.glob('./train/labels/*'): os.remove(f)

SOURCE = "./Scaled-Yolov4/ori_img/"
count_img = 0

for img_file in os.listdir(SOURCE + "images"):
    if(img_file[:2]!= "f_"): continue
    img = np.array(Image.open(SOURCE + "images/" + img_file))
    H, W, C = img.shape
    if C==4: img = img[:,:,:3]
    print(SOURCE + "images/" + img_file, img.shape)
    
    # Mỗi ảnh nhãn tạo 5 ảnh train
    for j in range(5*count_img, 5*(count_img + 1)):
        
        # Tạo màu nền trắng - xanh đậm - xanh nhạt
        output_img = 255 * np.ones((H, W, 3), np.uint8)
        if(random.randint(0, 4) == 1): output_img[:] = (24,97,121)
        elif(random.randint(0, 4) == 2): output_img[:] = (30,120,150)
        elif(random.randint(0, 4) == 3): output_img[:] = (26,108,135)

        new_label_txt = open('./train/labels/'+str(j)+'.txt', "w")
        label_txt = open(SOURCE + "labels/"+ img_file[:-4] + ".txt")
    
        for line in label_txt:
            # Lấy bbox x1,y1,x2,y2 của các object  trong ảnh nhãn
            line = line.split()
            x,y,w,h=float(line[1]),float(line[2]),int(float(line[3])*W),int(float(line[4])*H)
            x1 = int(x*W - w/2)
            y1 = int(y*H - h/2)
            x2 = x1 + w
            y2 = y1 + h
            crop = img[y1:y2, x1:x2, :]
            
            # Mỗi object được đem đi cắt/paste 2-6 lần
            frequence = random.randint(2, 6)
            for i in range(frequence):
                # Xoay và chọn ngẫu nhiên vị trí trong ảnh output
                crop_2 = ndimage.rotate(crop, angle=random.randint(0,180), cval=255)
                x1_, y1_ = random.randint(0, W - crop_2.shape[1]), random.randint(0, H - crop_2.shape[0])
                x2_ = x1_ + crop_2.shape[1]
                y2_ = y1_ + crop_2.shape[0]
                
                # Xoá viền răng cưa bằng cách chuyển các pixel hơi trắng thành trắng hẳn
                crop_2[crop_2.sum(axis=2) > 330] = (255,255,255)

                # Đến đây thì crop_2 là 1 hình chữ nhận nền trắng 255,255,255 + object
                # Khi Paste chỉ paste vùng có giá trị không trắng trong crop_2
                output_img[y1_:y2_, x1_:x2_, :][crop_2[:,:,:] < 255] = crop_2[crop_2[:,:,:] < 255]
                
                # Lưu nhãn
                new_label_txt.write(line[0]+" " + str((x1_+x2_)/2/W) + " " + 
                    str((y1_+y2_)/2/H) + " " + str(crop_2.shape[1]/W) + " " + 
                    str(crop_2.shape[0]/H) + "\n")
        
        Image.fromarray(output_img).save('./train/images/'+str(j)+'.png')   
        new_label_txt.close()
    
    count_img +=1
print("Create Traningset Done!")