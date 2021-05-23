from PIL import Image 
import glob, os, cv2
import numpy as np
import random, glob

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
SOURCE = "./Scaled-Yolov4/ori_img/"

noise1 = cv2.imread("./Scaled-Yolov4/noise1.png")
noise1 = cv2.cvtColor(noise1, cv2.COLOR_BGR2RGB)
noise2 = cv2.imread("./Scaled-Yolov4/noise2.png")
noise2 = cv2.cvtColor(noise2, cv2.COLOR_BGR2RGB)
noise3 = cv2.imread("./Scaled-Yolov4/noise3.png")
noise3 = cv2.cvtColor(noise3, cv2.COLOR_BGR2RGB)
ran_noise = random.randint(15, 50)

### GENERATE TRAINING DATA ###
for f in glob.glob('./train/labels.cache'): os.remove(f)
for f in glob.glob('./train/images/*'): os.remove(f)
for f in glob.glob('./train/labels/*'): os.remove(f)

count_img = 0

for img_file in os.listdir(SOURCE + "images"):
    if(img_file[:2]!= "f_"): continue
    img = np.array(Image.open(SOURCE + "images/" + img_file))
    H, W, C = img.shape
    if C==4: img = img[:,:,:3]
    print(SOURCE + "images/" + img_file, img.shape)
    
    for j in range(100*count_img, 100*(count_img + 1)):
        output_img = 255 * np.ones((H, W, 3), np.uint8)
        if(random.randint(0, 3) == 1): output_img[:] = (20,84,104)
        elif(random.randint(0, 3) == 2): output_img[:] = (30,119,148)
        elif(random.randint(0, 3) == 3): output_img[:] = (23,92,115)
        new_label_txt = open('./train/labels/'+str(j)+'.txt', "w")
        label_txt = open(SOURCE + "labels/"+ img_file[:-4] + ".txt")
        
    
        for line in label_txt:
            line = line.split()
            x,y,w,h=float(line[1]),float(line[2]),int(float(line[3])*W),int(float(line[4])*H)
            x1 = int(x*W - w/2)
            y1 = int(y*H - h/2)
            x2 = x1 + w
            y2 = y1 + h
            
            frequence = random.randint(2, 6)
            for i in range(frequence):
                # Toạ độ mới
                empty = 0
                count_failed = 0
                
                while empty == 0: 
                    count_failed += 1
                    if(count_failed > 100): break
                    crop = img[y1:y2, x1:x2, :]
                    crop_2 = ndimage.rotate(crop, angle=random.randint(0,180), cval=255)
                    x1_, y1_ = random.randint(0, W - crop_2.shape[1]), random.randint(0, H - crop_2.shape[0])
                    x2_ = x1_ + crop_2.shape[1]
                    y2_ = y1_ + crop_2.shape[0]
                    crop_2[crop_2.sum(axis=2) > 330] = (255,255,255)
                    # Nếu là giọt Drop thì cho phép paste luôn không cần xem xét là có đang đè không
                    empty = 1
#                     if(line[0] == '3'): empty = 1
#                     if (len(np.unique(output_img[y1_:y2_, x1_:x2_, :])) == 1): empty = 1
                        
                if(empty == 1):
                    output_img[y1_:y2_, x1_:x2_, :][crop_2[:,:,:] < 180] = crop_2[crop_2[:,:,:] < 180]
                    
                    new_label_txt.write(line[0]+" " + str((x1_+x2_)/2/W) + " " + 
                        str((y1_+y2_)/2/H) + " " + str(crop_2.shape[1]/W) + " " + 
                        str(crop_2.shape[0]/H) + "\n")
        
        
        ############################################################################################
        ############################################################################################
        # Paste noise:
        for i in range(ran_noise):
            noise1_ = ndimage.rotate(noise1, angle=random.randint(0,180), cval=255)
            empty = 0
            count_failed = 0
            while empty == 0: 
                count_failed += 1
                if(count_failed > 20): break
                x1_, y1_ = random.randint(0, W - noise1_.shape[1]), random.randint(0, H - noise1_.shape[0])
                x2_ = x1_ + noise1_.shape[1]
                y2_ = y1_ + noise1_.shape[0]
                if (len(np.unique(output_img[y1_:y2_, x1_:x2_, :])) == 1): empty = 1
            if(empty == 1):
                output_img[y1_:y2_, x1_:x2_, :][noise1_[:,:,0] < 255] = noise1_[noise1_[:,:,0] < 255]
            
            empty = 0
            count_failed = 0
            noise2_ = ndimage.rotate(noise2, angle=random.randint(0,180), cval=255)
            while empty == 0: 
                count_failed += 1
                if(count_failed > 20): break
                x1_, y1_ = random.randint(0, W - noise2_.shape[1]), random.randint(0, H - noise2_.shape[0])
                x2_ = x1_ + noise2_.shape[1]
                y2_ = y1_ + noise2_.shape[0]
                if (len(np.unique(output_img[y1_:y2_, x1_:x2_, :])) == 1): empty = 1
            if(empty == 1):
                output_img[y1_:y2_, x1_:x2_, :][noise2_[:,:,0] < 255] = noise2_[noise2_[:,:,0] < 255]
            
            empty = 0
            count_failed = 0
            noise3_ = ndimage.rotate(noise3, angle=random.randint(0,180), cval=255)
            while empty == 0: 
                count_failed += 1
                if(count_failed > 20): break
                x1_, y1_ = random.randint(0, W - noise3_.shape[1]), random.randint(0, H - noise3_.shape[0])
                x2_ = x1_ + noise3_.shape[1]
                y2_ = y1_ + noise3_.shape[0]
                if (len(np.unique(output_img[y1_:y2_, x1_:x2_, :])) == 1): empty = 1
            if(empty == 1):
                output_img[y1_:y2_, x1_:x2_, :][noise3_[:,:,0] < 255] = noise3_[noise3_[:,:,0] < 255]
        
#         print(max(np.unique(output_img)))
        Image.fromarray(output_img).save('./train/images/'+str(j)+'.png')   
#         mnmn = cv2.imread('./train/images/'+str(j)+'.png')
#         print(max(np.unique(output_img)))
        new_label_txt.close()
    
    count_img +=1
print("Create Traningset Done!")

### GENERATE VALIDATION SET ###

for f in glob.glob('./valid/labels.cache'): os.remove(f)
for f in glob.glob('./valid/images/*'): os.remove(f)
for f in glob.glob('./valid/labels/*'): os.remove(f)
    
count_img = 0

for img_file in os.listdir(SOURCE + "images"):
    if(img_file[:2]!= "f_"): continue
    img = np.array(Image.open(SOURCE + "images/" + img_file))
    H, W, C = img.shape
    if C==4: img = img[:,:,:3]
    print(SOURCE + "images/" + img_file, img.shape)
    
    for j in range(20*count_img, 20*(count_img + 1)):
        output_img = 255 * np.ones((H, W, 3), np.uint8)
        if(random.randint(0, 3) == 1): output_img[:] = (20,84,104)
        elif(random.randint(0, 3) == 2): output_img[:] = (30,119,148)
        elif(random.randint(0, 3) == 3): output_img[:] = (23,92,115)
        new_label_txt = open('./valid/labels/'+str(j)+'.txt', "w")
        label_txt = open(SOURCE + "labels/"+ img_file[:-4] + ".txt")
        
        
        for line in label_txt:
            line = line.split()
            x,y,w,h=float(line[1]),float(line[2]),int(float(line[3])*W),int(float(line[4])*H)
            x1 = int(x*W - w/2)
            y1 = int(y*H - h/2)
            x2 = x1 + w
            y2 = y1 + h
            
            frequence = random.randint(2, 6)
            for i in range(frequence):
                # Toạ độ mới
                empty = 0
                count_failed = 0
                
                while empty == 0: 
                    count_failed += 1
                    if(count_failed > 100): break
                    crop = img[y1:y2, x1:x2, :]
                    crop_2 = ndimage.rotate(crop, angle=random.randint(0,180), cval=255)
                    x1_, y1_ = random.randint(0, W - crop_2.shape[1]), random.randint(0, H - crop_2.shape[0])
                    x2_ = x1_ + crop_2.shape[1]
                    y2_ = y1_ + crop_2.shape[0]
                    crop_2[crop_2.sum(axis=2) > 330] = (255,255,255)
                    # Nếu là giọt Drop thì cho phép paste luôn không cần xem xét là có đang đè không
                    empty = 1
#                     if(line[0] == '3'): empty = 1
#                     if (len(np.unique(output_img[y1_:y2_, x1_:x2_, :])) == 1): empty = 1
                        
                if(empty == 1):
                    output_img[y1_:y2_, x1_:x2_, :][crop_2[:,:,:] < 200] = crop_2[crop_2[:,:,:] < 200]
                    
                    new_label_txt.write(line[0]+" " + str((x1_+x2_)/2/W) + " " + 
                        str((y1_+y2_)/2/H) + " " + str(crop_2.shape[1]/W) + " " + 
                        str(crop_2.shape[0]/H) + "\n")
        
        
        ############################################################################################
        ############################################################################################
        # Paste noise:
        for i in range(ran_noise):
            noise1_ = ndimage.rotate(noise1, angle=random.randint(0,180), cval=255)
            empty = 0
            count_failed = 0
            while empty == 0: 
                count_failed += 1
                if(count_failed > 20): break
                x1_, y1_ = random.randint(0, W - noise1_.shape[1]), random.randint(0, H - noise1_.shape[0])
                x2_ = x1_ + noise1_.shape[1]
                y2_ = y1_ + noise1_.shape[0]
                if (len(np.unique(output_img[y1_:y2_, x1_:x2_, :])) == 1): empty = 1
            if(empty == 1):
                output_img[y1_:y2_, x1_:x2_, :][noise1_[:,:,0] < 255] = noise1_[noise1_[:,:,0] < 255]
            
            empty = 0
            count_failed = 0
            noise2_ = ndimage.rotate(noise2, angle=random.randint(0,180), cval=255)
            while empty == 0: 
                count_failed += 1
                if(count_failed > 20): break
                x1_, y1_ = random.randint(0, W - noise2_.shape[1]), random.randint(0, H - noise2_.shape[0])
                x2_ = x1_ + noise2_.shape[1]
                y2_ = y1_ + noise2_.shape[0]
                if (len(np.unique(output_img[y1_:y2_, x1_:x2_, :])) == 1): empty = 1
            if(empty == 1):
                output_img[y1_:y2_, x1_:x2_, :][noise2_[:,:,0] < 255] = noise2_[noise2_[:,:,0] < 255]
            
            empty = 0
            count_failed = 0
            noise3_ = ndimage.rotate(noise3, angle=random.randint(0,180), cval=255)
            while empty == 0: 
                count_failed += 1
                if(count_failed > 20): break
                x1_, y1_ = random.randint(0, W - noise3_.shape[1]), random.randint(0, H - noise3_.shape[0])
                x2_ = x1_ + noise3_.shape[1]
                y2_ = y1_ + noise3_.shape[0]
                if (len(np.unique(output_img[y1_:y2_, x1_:x2_, :])) == 1): empty = 1
            if(empty == 1):
                output_img[y1_:y2_, x1_:x2_, :][noise3_[:,:,0] < 255] = noise3_[noise3_[:,:,0] < 255]
        
#         print(max(np.unique(output_img)))
        Image.fromarray(output_img).save('./valid/images/'+str(j)+'.png')   
#         mnmn = cv2.imread('./train/images/'+str(j)+'.png')
#         print(max(np.unique(output_img)))
        new_label_txt.close()
    
    count_img +=1
print("Create Validation Done!")