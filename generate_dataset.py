from PIL import Image 
import glob, os, cv2
import numpy as np
import random, glob, csv, json
from scipy import ndimage
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_num', type=int, default=1000, help='Number of traning images')
parser.add_argument('--valid_num', type=int, default=200, help='Number of validation images')
parser.add_argument('--num_object_max', type=int, default=1100, help='Maximum number of objects per images')
parser.add_argument('--num_object_min', type=int, default=900, help='Minimum number of objects per images')
parser.add_argument('--thresh_pixel', type=int, default=350, help='Sum pixel threshold to avoid object border errors when cropping')
parser.add_argument('--source', type=str, default="./data/data_1/", help='path to source images folder')
parser.add_argument('--background', type=str, default="./data/background_1/", help='path to source background folder')
parser.add_argument('--gray', action='store_true', help='Gray or RGB image')
parser.add_argument('--empty', action='store_true', help='Empty train/valid folder or not')
parser.add_argument('--height', type=int, default=1200, help='Height of images')
parser.add_argument('--width', type=int, default=1200, help='Width of images')
opt = parser.parse_args()
print(opt)
    
### GENERATE DATASET ###
def generate_dataset(mode="train"):
    if (mode=="train"):
        image_folder = './train/images/'
        label_folder = './train/labels/'
        num_data = opt.train_num
    elif (mode=="valid"):
        image_folder = './valid/images/'
        label_folder = './valid/labels/'
        num_data = opt.valid_num
    
    print("Generating " + mode + " dataset ...")
    count_img = len([name for name in os.listdir(image_folder) if os.path.isfile(name)])
    for j in range(count_img, num_data + count_img):
        if(j%50==0): print("---", j, "images")
            
        ignored = {".ipynb_checkpoints", ".DS_Store"}
        folders = [x for x in os.listdir(opt.background) if x not in ignored]
        output_img = random.choice(folders)
#         print(output_img)
        output_img = np.array(Image.open(os.path.join(opt.background, output_img)), dtype=np.uint8)

        new_label_txt = open(label_folder+str(j)+'.txt', "w")
        frequence = random.randint(opt.num_object_min, opt.num_object_max) # Số lượng object mỗi ảnh được gen ra

        for i in range(frequence):
            random_object = random.randint(0, len(img_pool) - 1)
            crop = img_pool[random_object]
            crop = ndimage.rotate(crop, angle=random.randint(0,180), cval=255)
            x1_, y1_ = random.randint(0, opt.width - crop.shape[1]), random.randint(0, opt.height - crop.shape[0])
            x2_ = x1_ + crop.shape[1]
            y2_ = y1_ + crop.shape[0]
            crop[crop.sum(axis=2) > opt.thresh_pixel] = (255,255,255)
            
            output_img[y1_:y2_, x1_:x2_, :][crop[:,:,:] < 255] = crop[crop[:,:,:] < 255]
            new_label_txt.write(label_pool[random_object]+" " + str((x1_+x2_)/2/opt.width) + " " + 
                    str((y1_+y2_)/2/opt.height) + " " + str(crop.shape[1]/opt.width) + " " + 
                    str(crop.shape[0]/opt.height) + "\n")

        if opt.gray:
            # Convert to gray then to 3 gray channels img
            output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2GRAY) 
            output_img = np.stack((output_img,)*3, axis=-1)
        Image.fromarray(output_img).save(image_folder+str(j)+'.png')   
        new_label_txt.close()
        count_img +=1
    print("Create " + mode + " data done!")

if __name__ == '__main__':
    
    if (opt.empty):
        for f in glob.glob('./train/labels.cache'): os.remove(f)
        for f in glob.glob('./train/images/*'): os.remove(f)
        for f in glob.glob('./train/labels/*'): os.remove(f)

        for f in glob.glob('./valid/labels.cache'): os.remove(f)
        for f in glob.glob('./valid/images/*'): os.remove(f)
        for f in glob.glob('./valid/labels/*'): os.remove(f)
        
    ### Create lable and image pool
    img_pool = []
    label_pool = []
    for img_file in os.listdir(os.path.join(opt.source, "images")):
        img = np.array(Image.open(os.path.join(os.path.join(opt.source, "images"), img_file)))
        H, W, C = img.shape
        if C==4: img = img[:,:,:3]
        label_txt = open(os.path.join(opt.source, "labels/"+ img_file[:-4] + ".txt"))
        print("Reading", os.path.join(os.path.join(opt.source, "images"), img_file), img.shape)

        # Read label txt file then crop object from image
        for line in label_txt:
            line = line.split()
            label_pool.append(str(line[0]))
            x,y,w,h=float(line[1]),float(line[2]),int(float(line[3])*W),int(float(line[4])*H)
            x1 = int(x*W - w/2)
            y1 = int(y*H - h/2)
            x2 = x1 + w
            y2 = y1 + h
            crop = img[y1:y2, x1:x2, :]
            img_pool.append(crop)
    # print(len(img_pool), len(label_pool))
    
    ### Generate data
    generate_dataset(mode="train")
    generate_dataset(mode="valid")