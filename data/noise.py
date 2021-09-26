from PIL import Image 
import glob, os, cv2
import numpy as np
import random, glob, csv, json
from scipy import ndimage
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--num_object_max', type=int, default=1100, help='Maximum number of noise objects per images')
parser.add_argument('--num_object_min', type=int, default=900, help='Minimum number of noise objects per images')
parser.add_argument('--height', type=int, default=1200, help='Height of images')
parser.add_argument('--width', type=int, default=1200, help='Width of images')
opt = parser.parse_args()
print(opt)


ran_noise = random.randint(opt.num_object_min, opt.num_object_max)

noise1 = cv2.imread("./noise/noise1.png")
noise1 = cv2.cvtColor(noise1, cv2.COLOR_BGR2RGB)
noise2 = cv2.imread("./noise/noise2.png")
noise2 = cv2.cvtColor(noise2, cv2.COLOR_BGR2RGB)
noise3 = cv2.imread("./noise/noise3.png")
noise3 = cv2.cvtColor(noise3, cv2.COLOR_BGR2RGB)

noise4 = cv2.imread("./noise/noise4.png")
noise4 = cv2.cvtColor(noise4, cv2.COLOR_BGR2RGB)
noise5 = cv2.imread("./noise/noise5.png")
noise5 = cv2.cvtColor(noise5, cv2.COLOR_BGR2RGB)
noise6 = cv2.imread("./noise/noise6.png")
noise6 = cv2.cvtColor(noise6, cv2.COLOR_BGR2RGB)

def paste_noise(output_img):
    for i in range(ran_noise):
        noise1_ = ndimage.rotate(noise1, angle=random.randint(0,180), cval=255)
        empty = 0
        count_failed = 0
        while empty == 0: 
            count_failed += 1
            if(count_failed > 20): break
            x1_, y1_ = random.randint(0, opt.width - noise1_.shape[1]), random.randint(0, opt.height - noise1_.shape[0])
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
            x1_, y1_ = random.randint(0, opt.width - noise2_.shape[1]), random.randint(0, opt.height - noise2_.shape[0])
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
            x1_, y1_ = random.randint(0, opt.width - noise3_.shape[1]), random.randint(0, opt.height - noise3_.shape[0])
            x2_ = x1_ + noise3_.shape[1]
            y2_ = y1_ + noise3_.shape[0]
            if (len(np.unique(output_img[y1_:y2_, x1_:x2_, :])) == 1): empty = 1
        if(empty == 1):
            output_img[y1_:y2_, x1_:x2_, :][noise3_[:,:,0] < 255] = noise3_[noise3_[:,:,0] < 255] 
            
    return output_img

def paste_noise_2(output_img):
    for i in range(ran_noise):
        noise1_ = ndimage.rotate(noise1, angle=random.randint(0,180), cval=255)
        x1_, y1_ = random.randint(0, opt.width - noise1_.shape[1]), random.randint(0, opt.height - noise1_.shape[0])
        x2_ = x1_ + noise1_.shape[1]
        y2_ = y1_ + noise1_.shape[0]
        output_img[y1_:y2_, x1_:x2_, :][noise1_[:,:,0] < 255] = noise1_[noise1_[:,:,0] < 255]

        noise2_ = ndimage.rotate(noise2, angle=random.randint(0,180), cval=255)
        x1_, y1_ = random.randint(0, opt.width - noise2_.shape[1]), random.randint(0, opt.height - noise2_.shape[0])
        x2_ = x1_ + noise2_.shape[1]
        y2_ = y1_ + noise2_.shape[0]
        output_img[y1_:y2_, x1_:x2_, :][noise2_[:,:,0] < 255] = noise2_[noise2_[:,:,0] < 255]

        noise3_ = ndimage.rotate(noise3, angle=random.randint(0,180), cval=255)
        x1_, y1_ = random.randint(0, opt.width - noise3_.shape[1]), random.randint(0, opt.height - noise3_.shape[0])
        x2_ = x1_ + noise3_.shape[1]
        y2_ = y1_ + noise3_.shape[0]
        output_img[y1_:y2_, x1_:x2_, :][noise3_[:,:,0] < 255] = noise3_[noise3_[:,:,0] < 255] 
            
    return output_img

def paste_noise_3(output_img):
    for i in range(ran_noise):
        noise4_ = ndimage.rotate(noise4, angle=random.randint(0,180), cval=255)
        x1_, y1_ = random.randint(0, opt.width - noise4_.shape[1]), random.randint(0, opt.height - noise4_.shape[0])
        x2_ = x1_ + noise4_.shape[1]
        y2_ = y1_ + noise4_.shape[0]
        output_img[y1_:y2_, x1_:x2_, :][noise4_[:,:,0] < 255] = noise4_[noise4_[:,:,0] < 255]

        noise5_ = ndimage.rotate(noise5, angle=random.randint(0,180), cval=255)
        x1_, y1_ = random.randint(0, opt.width - noise5_.shape[1]), random.randint(0, opt.height - noise5_.shape[0])
        x2_ = x1_ + noise5_.shape[1]
        y2_ = y1_ + noise5_.shape[0]
        output_img[y1_:y2_, x1_:x2_, :][noise5_[:,:,0] < 255] = noise5_[noise5_[:,:,0] < 255]

        noise6_ = ndimage.rotate(noise6, angle=random.randint(0,180), cval=255)
        x1_, y1_ = random.randint(0, opt.width - noise6_.shape[1]), random.randint(0, opt.height - noise6_.shape[0])
        x2_ = x1_ + noise6_.shape[1]
        y2_ = y1_ + noise6_.shape[0]
        output_img[y1_:y2_, x1_:x2_, :][noise6_[:,:,0] < 255] = noise6_[noise6_[:,:,0] < 255] 
            
    return output_img

def create_label_pool_from_custom(img_pool, label_pool):
    img_gold = np.array(Image.open("./DATASET/picture.png"))
    if (img_gold.shape[2] ==4): img_gold = img_gold[:,:,:3]
    csv_file = open('./DATASET/label.csv', mode='r')
    csv_reader = csv.DictReader(csv_file)
    totalrows = sum(1 for _ in open('./DATASET/label.csv'))
    
    line_count = 0
    mult = 1.2
    for row in csv_reader:
        line_count += 1
        data = json.loads(row["region_shape_attributes"])
        
        try:
            if (data["name"] == "rect"):
                small = img_gold[data["y"]:(data["y"]+data["height"]), data["x"]:(data["x"]+data["width"])]
                img_pool.append(small)
                label_pool.append(str(data["type"]))

            if (data["name"] == "polygon"):
                polygon = []
                for j in range(len(data["all_points_x"])):
                    polygon.append([data["all_points_x"][j], data["all_points_y"][j]])
                
                # Cắt ra hình chữ nhật tối thiểu bao quanh polygon
                rect = cv2.minAreaRect(np.array([polygon]))
                box = np.int0(cv2.boxPoints(rect))
                opt.width = rect[1][0]
                opt.height = rect[1][1]
                Xs = [i[0] for i in box]
                Ys = [i[1] for i in box]
                x1 = min(Xs)
                x2 = max(Xs)
                y1 = min(Ys)
                y2 = max(Ys)
                rotated = False
                angle = rect[2]
                if angle < -45:
                    angle+=90
                    rotated = True
                croppedW = opt.width if not rotated else opt.height 
                croppedH = opt.height if not rotated else opt.width
                center = (int((x1+x2)/2), int((y1+y2)/2))
                size = (int(mult*(x2-x1)),int(mult*(y2-y1)))
                M = cv2.getRotationMatrix2D((size[0]/2, size[1]/2), angle, 1.0)

                cropped = cv2.warpAffine(cv2.getRectSubPix(img_gold, size, center), M, size)
                small = cv2.getRectSubPix(cropped, (int(croppedW*mult), int(croppedH*mult)), (size[0]/2, size[1]/2))
        
                img_pool.append(small)
                label_pool.append(str(data["type"]))
        except:
            print("------------ Có lỗi!!")
    return img_pool, label_pool

if __name__ == '__main__':
    
    # generate bg for type 1
    output_img = 255 * np.ones((opt.height, opt.width, 3), np.uint8)
    Image.fromarray(output_img).save('./background_1/1.png')   
    
    output_img[:] = (24,97,121)
    Image.fromarray(output_img).save('./background_1/2.png')   
    
    output_img[:] = (30,120,150)
    Image.fromarray(output_img).save('./background_1/3.png')   
    
    output_img[:] = (26,108,135)
    Image.fromarray(output_img).save('./background_1/4.png')   
    
    output_img = 255 * np.ones((opt.height, opt.width, 3), np.uint8)
    output_img = paste_noise_2(output_img)
    Image.fromarray(output_img).save('./background_1/5.png')   
    
    # generate bg for type 2
    output_img = 255 * np.ones((opt.height, opt.width, 3), np.uint8)
    Image.fromarray(output_img).save('./background_2/1.png')   
    
    output_img[:] = (111,217,0)
    Image.fromarray(output_img).save('./background_2/2.png')   
    
    output_img[:] = (92,182,0)
    Image.fromarray(output_img).save('./background_2/3.png') 
    
    output_img[:] = (97,237,0)
    Image.fromarray(output_img).save('./background_2/4.png')   
    
    output_img = 255 * np.ones((opt.height, opt.width, 3), np.uint8)
    output_img = paste_noise_3(output_img)
    Image.fromarray(output_img).save('./background_2/5.png')   
    