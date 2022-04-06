import random
import numpy as np
import os
import torch
from torch.utils.data import Dataset
import rawpy
import glob
import imageio
import matplotlib.pyplot as plt
import cv2

def define_weights(num):
    weights = np.float32((np.logspace(0,num,127, endpoint=True, base=10.0)))
    weights = weights/np.max(weights)
    weights = np.flipud(weights).copy()    
    return weights

def get_na(bins,weights,img_loww,amp=1.0):
    H,W = img_loww.shape
    arr = img_loww*1
    selection_dict = {weights[0]: (bins[0]<=arr)&(arr<bins[1])}
    for ii in range(1,len(weights)):
        selection_dict[weights[ii]] = (bins[ii]<=arr)&(arr<bins[ii+1])
    mask = np.select(condlist=selection_dict.values(), choicelist=selection_dict.keys())
   
    mask_sum1 = np.sum(mask,dtype=np.float64)
    
    na1 = np.float32(np.float64(mask_sum1*0.01*amp)/np.sum(img_loww*mask,dtype=np.float64))

    if na1>300.0:
        na1 = np.float32(300.0)
    if na1<1.0:
        na1 = np.float32(1.0)
    
    selection_dict.clear()

    return na1



def color_pixels(raw_color_index):
    red     = np.array(raw_color_index==0)
    green1  = np.array(raw_color_index==1)
    blue    = np.array(raw_color_index==2)
    green2  = np.array(raw_color_index==3)
    green   = green1 | green2
    return red, green1, blue, green2


def scale(x, min, max, range=1):
    if x < min:
        x = min
    if x > max:
        x = max
    scaled = (x-min)/(max-min)*range
    return scaled

def scale_array(x, min, max, range=1):
    x[x < min] = min
    x[x > max] = max
    scaled = (x-min)/(max-min)*range
    return scaled

def descale_array(scaled, min, max, range=1):
    # color = (bayer-min)/(max-min)*range
    x = scaled*(max-min)/range #+min
    return x


def bayer2rgb(x, min, max, range, wb):
    scaled_color = scale(x, min, max, range)
    true_color = scaled_color*wb
    return int(true_color)

def bayer2rgb_array(x, min, max, range, wb):
    scaled_color = scale_array(x, min, max, range)
    true_color = scaled_color*wb
    return true_color.astype(int)

def rgb2bayer_array(true_color, min, max, range, wb):
    scaled_color = true_color/wb
    x = descale_array(scaled_color, min, max, range)
    return x


# defines a bunch of parameters which (can) help with the conversion from bayer to rgb. Can still be improved.
def set_parameters(raw, img):
    rgb2xyz_matrix = raw.rgb_xyz_matrix
    xyz2rgb_matrix = np.linalg.inv(rgb2xyz_matrix[[0,1,2],:])

    max_color = np.amax(img)
    min_color = np.amin(img)
    average_color = np.average(img)

    sat_vals = raw.camera_white_level_per_channel
    sat_r, sat_g1, sat_b, sat_g2 = sat_vals

    white_balance = np.array(raw.camera_whitebalance)
    wb_r, wb_g1, wb_b, wb_g2 = white_balance/max(white_balance)
    wb_g = (wb_g1 + wb_g2)/2
    wb_r, wb_g, wb_b = 1, 1, 1                # 1.5, 0.8, 1
    wb = [wb_r, wb_g, wb_b] # white balance of different colors

    r_min, r_max = 512, 523.3
    g_min, g_max = 512, 535
    b_min, b_max = 512, 520.3
    cmin = [r_min, g_min, b_min] # minimimum (=dark) pixel value in bayer image 
    cmax = [r_max, g_max, b_max] # maximum (=saturated) pixel value in bayer image

    # img_loww = (np.maximum(img - 512,0)/(16383 - 512))
    
    return wb, cmin, cmax

def add_noise(color, scale):
    shape = color.shape
    noise = np.random.normal(loc=0.0, scale=scale, size=shape)
    color = np.add(color, noise).astype(int)
    return color


def change_brightness(color, amount):
    color = np.add(color, amount)
    return color


def add_bayer(bayer, color_index, amount):
    amount = amount.flatten()
    bayer[color_index] = bayer[color_index]+amount
    return bayer

def rgbchanges2bayer(rgb_change, img, params):
    r_changes = rgb_change[:,:,0]
    g_changes = rgb_change[:,:,1]
    b_changes = rgb_change[:,:,2]

    wb, cmin, cmax = params
    wb_r, wb_g, wb_b = wb
    r_min, g_min, b_min = cmin
    r_max, g_max, b_max = cmax

    r_bayer_changes = rgb2bayer_array(r_changes, r_min, r_max, 255, wb_r)
    g_bayer_changes = rgb2bayer_array(g_changes, g_min, g_max, 255, wb_g)
    b_bayer_changes = rgb2bayer_array(b_changes, b_min, b_max, 255, wb_b)

    bayer_changes = [r_bayer_changes, g_bayer_changes, b_bayer_changes]

    return bayer_changes

def apply_bounds(rgb):
    rgb[rgb<0] = 0
    rgb[rgb>255] = 0
    return rgb


def part_init(train_files):

    bins = np.float32((np.logspace(0,8,128, endpoint=True, base=2.0)-1))/255.0
    weights5 = define_weights(5)
    train_list = []
    
    for i in range(len(train_files)):

        #----------------------------
        # EXTRACT DATA FROM RAW FILE
        #----------------------------
        raw = rawpy.imread(train_files[i])
        img = raw.raw_image_visible.astype(np.float32).copy()
        
        ### this works, but brakes the rest of the code. Only use as reference!!!
        # raw_parameters = rawpy.Params(use_camera_wb=True)
        # rgb_postprocess = raw.postprocess(params=raw_parameters) # use this to get the exact rgb values (and accurate image), but it can't be reconverted to bayer
        # plt.imshow(rgb_postprocess)
        # plt.show()

        # info2 = raw.sizes
        # print(info2)
        # info3 = raw.tone_curve
        # print(info3)
        # info4 = raw.white_level
        # print(info4)
        # info5 = raw.raw_type
        # print(info5)
        # info6 = raw.raw_colors_visible
        # print(info6)
        # info7 = raw.num_colors
        # print(info7)
        # info8 = raw.color_desc   #0123 = RGBG
        # print(info8)
        # info9 = raw.camera_white_level_per_channel
        # print(info9)
        # info10 = raw.camera_whitebalance
        # print(info10)
        # info11 = raw.color_matrix
        # print(info11)

        
        #------------------------------
        # CONVERT BAYER INTO R, G and B
        #------------------------------
        params = set_parameters(raw, img)
        wb, cmin, cmax = params
        wb_r, wb_g, wb_b = wb
        r_min, g_min, b_min = cmin
        r_max, g_max, b_max = cmax

        raw_color_index = raw.raw_colors_visible
        color_indexes = color_pixels(raw_color_index)
        red, green1, blue, green2 = color_indexes

        rgb_shape = (int(img.shape[0]/2),int(img.shape[1]/2))

        r  = np.reshape(img[red],    rgb_shape)
        g1 = np.reshape(img[green1], rgb_shape)
        g2 = np.reshape(img[green2], rgb_shape)
        b  = np.reshape(img[blue],   rgb_shape)
        g = (g1+g2)/2

        r = bayer2rgb_array(r, r_min, r_max, range=255, wb=wb_r)
        g = bayer2rgb_array(g, g_min, g_max, range=255, wb=wb_g)
        b = bayer2rgb_array(b, b_min, b_max, range=255, wb=wb_b)

        # rgb = np.zeros((int(img.shape[0]/2),int(img.shape[1]/2),3))
        # for h in range(len(rgb)):
        #     for w in range(len(rgb[0])):
        #         r = img[2*h,2*w]
        #         g1 = img[2*h,2*w+1]
        #         b = img[2*h+1,2*w+1]
        #         g2 = img[2*h+1,2*w]
        #         g = (g1+g2)/2

        #         r = bayer2rgb(r, r_min, r_max, range=255, wb=wb_r)
        #         g = bayer2rgb(g, g_min, g_max, range=255, wb=wb_g)
        #         b = bayer2rgb(b, b_min, b_max, range=255, wb=wb_b)
                
        #         pixel = [r,g,b]
        #         rgb[h,w] = pixel

        # rgb = rgb.astype(int)

        rgb_original = cv2.merge([r, g, b])
        rgb_original = apply_bounds(rgb_original)

        raw_parameters = rawpy.Params(use_camera_wb=True)
        rgb_postprocess = raw.postprocess(params=raw_parameters) # use this to get the exact rgb values

        r22 = rgb_postprocess[:,:,0]
        g22 = rgb_postprocess[:,:,1]
        b22 = rgb_postprocess[:,:,2]

        print(np.average(r), np.average(r22))
        print(np.average(g), np.average(g22))
        print(np.average(b), np.average(b22))

        # print('postproces ', rgb_postprocess)
        # print('converted ', rgb_original)

        f = plt.figure()
        f.add_subplot(1,2,1)
        plt.imshow(rgb_postprocess)
        f.add_subplot(1,2, 2)
        plt.imshow(rgb_original)
        plt.show()

        #---------------------------------------------
        # DO OPERATIONS ON R, G and B color spectrums
        #---------------------------------------------

        # change_brightness, add_noise,...
           
        # r = add_noise(r, 30)

        r = change_brightness(r, 1) # change the red brightness

        # b = change_brightness(b, 50) # change the blue brightness

        #---------------------------------------------
        # SHOW RESULT
        #---------------------------------------------
        rgb = cv2.merge([r, g, b])
        rgb = apply_bounds(rgb) # to prevent changed numbers being smaller than 0 or larger than 255  

        # plt.imshow(rgb)
        # plt.show()      

        #---------------------------------------------
        # APPLY CHANGES TO BAYER, CONVERT RGB TO BAYER
        #---------------------------------------------

        rgb_change = rgb - rgb_original
        bayer_changes = rgbchanges2bayer(rgb_change, img, params)

        r_bayer_changes, g_bayer_changes, b_bayer_changes = bayer_changes
        red_index, green1_index, blue_index, green2_index = color_indexes

        img = add_bayer(img, red_index, r_bayer_changes)
        img = add_bayer(img, green1_index, g_bayer_changes)
        img = add_bayer(img, blue_index, b_bayer_changes)
        img = add_bayer(img, green2_index, g_bayer_changes)

        raw.close()
        
        h,w = img.shape
        if h%32!=0:
            print('Image dimensions should be multiple of 32. Correcting the 1st dimension.')
            h = (h//32)*32
            img = img[:h,:]
        
        if w%32!=0:
            print('Image dimensions should be multiple of 32. Correcting the 2nd dimension.')
            w = (w//32)*32
            img = img[:,:w]
        
        img_loww = (np.maximum(img - 512,0)/ (16383 - 512))       

        na5 = get_na(bins,weights5,img_loww)   
        
        img_loww = img_loww*na5
             
        train_list.append(img_loww)

        print('Image No.: {}, Amplification_m=1: {}'.format(i+1,na5))
    return train_list
    
    
################ DATASET CLASS
class load_data(Dataset):
    """Loads the Data."""
    
    def __init__(self, train_files):    
        print('\n...... Loading all files to CPU RAM\n')
        self.train_list = part_init(train_files)        
        print('\nFiles loaded to CPU RAM......\n')
        
    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, idx):    
        img_low = self.train_list[idx]
        return torch.from_numpy(img_low).float().unsqueeze(0) 

def run_test(model, dataloader_test, save_images):    
    with torch.no_grad():
        model.eval()
        for image_num, low in enumerate(dataloader_test):
            low = low.to(next(model.parameters()).device)            
            for amp in [1.0,5.0,8.0, 15, 40]:
                pred = model(amp*low)
                pred = (np.clip(pred[0].detach().cpu().numpy().transpose(1,2,0),0,1)*255).astype(np.uint8)
                imageio.imwrite(os.path.join(save_images,'img_num_{}_m_{}.jpg'.format(image_num,amp)), pred)
    return
    
    

        
