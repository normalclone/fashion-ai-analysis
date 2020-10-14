#-*-coding:utf-8-*-
import torch
from collections import Counter
import numpy as np 
import argparse
import pickle 
import os
import time
from torchvision import transforms 
from model import EncoderClothing, DecoderClothing
from darknet import Darknet
from PIL import Image
from util import *
import cv2
import pickle as pkl
from preprocess import prep_image2

import sys
if sys.version_info >= (3,0):
    from roi_align.roi_align import RoIAlign
else :
    from roi_align import RoIAlign

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


color = {   'white':0, 'black':1, 'grey':2, 'pink':3, 'red':4,
            'green':5, 'blue':6, 'brown':7, 'navy':8, 'beige':9, 'yellow':10, 'purple':11,
            'orange':12, 'others':13 }

pattern = { 'single':0, 'checker':1, 'dotted':2, 'floral':3, 'striped':4, 'others':5 }

gender = { 'man':0, 'woman':1 }

season = { 'spring':0, 'summer':1, 'autumn':2, 'winter':3 }

c_class = { 'shirt':0, 'jumper':1, 'jacket':2, 'vest':3, 'coat':4,
            'dress':5, 'pants':6, 'skirt':7 }

sleeves = { 'long':0, 'short':1, 'no':2 }

a_class = { 'scarf':0, 'cane':1, 'bag':2, 'shoes':3, 'hat':4, 'face':5, 'glasses':6}

colors_a =     ["", "white", "black", "gray", "pink", "red", "green", "blue", "brown", "navy", "beige", \
    "yellow", "purple", "orange", "mixed color"] #0-14
pattern_a =    ["", "no pattern", "checker", "dotted", "floral", "striped", "custom pattern"] #0-6
gender_a =     ["", "man", "woman"] #0-2
season_a =     ["", "spring", "summer", "autumn", "winter"] #0-4
upper_t_a =    ["", "shirt", "jumper", "jacket", "vest", "parka", "coat", "dress"]#0-7
u_sleeves_a =  ["", "short sleeves", "long sleeves", "no sleeves"]#0-3

lower_t_a =    ["", "pants", "skirt"]#0-2
l_sleeves_a =  ["", "short", "long"]#0-2
leg_pose_a =   ["", "standing", "sitting", "lying"]#0-3

glasses_a =    ["", "glasses"]

attribute_pool = [colors_a, pattern_a, gender_a, season_a, upper_t_a, u_sleeves_a, \
                  colors_a, pattern_a, gender_a, season_a, lower_t_a, l_sleeves_a, leg_pose_a]

def counter_parts1(sentence):
   
    color_bin = np.zeros(14)
    pattern_bin = np.zeros(6)
    gender_bin = np.zeros(2)
    season_bin = np.zeros(4)
    class_bin = np.zeros(8)
    sleeves_bin = np.zeros(3)
            
    token = sentence.split()
    for word in token:
        c_pos = color.get(word)
        p_pos = pattern.get(word)
        g_pos = gender.get(word)
        s_pos = season.get(word)
        cl_pos = c_class.get(word)
        sl_pos = sleeves.get(word)

        if c_pos is not None:
            color_bin[c_pos] = 1
        elif p_pos is not None:
            pattern_bin[p_pos] = 1
        elif g_pos is not None:
            gender_bin[g_pos] = 1
        elif s_pos is not None:
            season_bin[s_pos] = 1
        elif cl_pos is not None:
            class_bin[cl_pos] = 1
        elif sl_pos is not None:
            sleeves_bin[sl_pos] = 1
            
    return color_bin, pattern_bin, gender_bin, season_bin, class_bin, sleeves_bin

def counter_parts2(sentence):
   
    color_bin = np.zeros(14)
    pattern_bin = np.zeros(6)
    gender_bin = np.zeros(2)
    season_bin = np.zeros(4)
    class_bin = np.zeros(8)
    sleeves_bin = np.zeros(3)
            
    token = sentence.split()
    for word in token:
        c_pos = color.get(word)
        p_pos = pattern.get(word)
        g_pos = gender.get(word)
        s_pos = season.get(word)
        cl_pos = c_class.get(word)
        sl_pos = sleeves.get(word)

        if c_pos is not None:
            color_bin[c_pos] = 1
        elif p_pos is not None:
            pattern_bin[p_pos] = 1
        elif g_pos is not None:
            gender_bin[g_pos] = 1
        elif s_pos is not None:
            season_bin[s_pos] = 1
        elif cl_pos is not None:
            class_bin[cl_pos] = 1
        elif sl_pos is not None:
            sleeves_bin[sl_pos] = 1
            
    return color_bin, pattern_bin, gender_bin, season_bin, class_bin, sleeves_bin


def keep_majority(sentence, color_stream, pattern_stream, gender_stream, season_stream, class_stream, sleeves_stream):

    tokens = sentence.split('and')
    color_bin, pattern_bin, gender_bin, season_bin, class_bin, sleeves_bin = counter_parts1(tokens[0]) # for tokens[0] ==> upper
    if tokens.__len__() == 2:
        color_bin, pattern_bin, gender_bin, season_bin, class_bin, sleeves_bin = counter_parts1(tokens[1]) # for tokens[1] ==> lower

    if color_stream.__len__() < 50:
        color_stream.insert(0, color_bin)
        pattern_stream.insert(0, pattern_bin)
        gender_stream.insert(0, gender_bin)
        season_stream.insert(0, season_bin)
        class_stream.insert(0, class_bin)
        sleeves_stream.insert(0, sleeves_bin)
        return ''
    else:
        color_stream.pop()
        pattern_stream.pop()
        gender_stream.pop()
        season_stream.pop()
        class_stream.pop()
        sleeves_stream.pop()
        
        color_stream.insert(0, color_bin)
        pattern_stream.insert(0, pattern_bin)
        gender_stream.insert(0, gender_bin)
        season_stream.insert(0, season_bin)
        class_stream.insert(0, class_bin)
        sleeves_stream.insert(0, sleeves_bin)

        #print(sum(season_stream))

        c_max = np.argmax(sum(color_stream))
        p_max = np.argmax(sum(pattern_stream))
        g_max = np.argmax(sum(gender_stream))
        s_max = np.argmax(sum(season_stream))
        cl_max = np.argmax(sum(class_stream))
        sl_max = np.argmax(sum(sleeves_stream))
        
        """ new_t = list()
        new_t.append([k for k, v in color.items() if v == c_max][0])
        if [k for k, v in pattern.items() if v == p_max][0] != 'single':
            new_t.append([k for k, v in pattern.items() if v == p_max][0])

        new_t.append([k for k, v in gender.items() if v == g_max][0])
        new_t.append([k for k, v in season.items() if v == s_max][0])
        new_t.append([k for k, v in c_class.items() if v == cl_max][0])
        new_t.append('with')
        new_t.append([k for k, v in sleeves.items() if v == sl_max][0])
        new_t.append('sleeves')

        new_sentence = '======> ' +  ' '.join(new_t) """

        new_t = list()
        #new_t.append([k for k, v in c_class.items() if v == cl_max][0])
        #new_t.append(':')
        new_t.append([k for k, v in color.items() if v == c_max][0])
        if [k for k, v in pattern.items() if v == p_max][0] != 'single':
            new_t.append([k for k, v in pattern.items() if v == p_max][0])

        new_t.append([k for k, v in gender.items() if v == g_max][0])
        new_t.append([k for k, v in season.items() if v == s_max][0])
        new_t.append([k for k, v in c_class.items() if v == cl_max][0])
        new_t.append('with')
        new_t.append([k for k, v in sleeves.items() if v == sl_max][0])
        new_t.append('sleeves')

        #new_sentence = '======> ' +  ' '.join(new_t)

        new_sentence = ' '.join(new_t)
        
        #print(sum(sleeves_stream))q
        #print(new_sentence)
        #sys.stdout.write(new_sentence + '\r\r')
        #sys.stdout.flush()
        return new_sentence

def main(args):
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor()])


    num_classes = 80
    yolov3  = Darknet(args.cfg_file)
    yolov3.load_weights(args.weights_file)
    yolov3.net_info["height"] = args.reso    
    inp_dim = int(yolov3.net_info["height"])
    assert inp_dim % 32 == 0 
    assert inp_dim > 32
    print("yolo-v3 network successfully loaded")

    attribute_size = [15, 7, 3, 5, 8, 4, 15, 7, 3, 5, 3, 3, 4]

    encoder = EncoderClothing(args.embed_size, device, args.roi_size, attribute_size)

    yolov3.to(device)
    encoder.to(device)

    yolov3.eval()
    encoder.eval()

    encoder.load_state_dict(torch.load(args.encoder_path))

    #cap = cv2.VideoCapture('demo2.mp4')
    
    cap = cv2.VideoCapture(0) 
    assert cap.isOpened(), 'Cannot capture source'
    
    frames = 0
    start = time.time() 

    counter = Counter()
    color_stream = list()
    pattern_stream = list()
    gender_stream = list()
    season_stream = list()
    class_stream = list()
    sleeves_stream = list()

    ret, frame = cap.read()
    if ret:
        
        image, orig_img, dim = prep_image2(frame, inp_dim)
        im_dim = torch.FloatTensor(dim).repeat(1,2)      
        
        image_tensor = image.to(device)
    detections = yolov3(image_tensor, device, True) 


    os.system('clear')
    cv2.imshow("frame", orig_img)
    cv2.moveWindow("frame", 50, 50)
    text_img = np.zeros((200, 1750, 3))
    cv2.imshow("text", text_img)
    cv2.moveWindow("text", 50, dim[1]+110)

    while cap.isOpened():
        
        ret, frame = cap.read()
        #### ret, frame = ros_message_cam_image()
        if ret:
            
            image, orig_img, dim = prep_image2(frame, inp_dim)
            im_dim = torch.FloatTensor(dim).repeat(1,2)      
            
            image_tensor = image.to(device)
            im_dim = im_dim.to(device)
            
            # Generate an caption from the image
            detections = yolov3(image_tensor, device, True) # prediction mode for yolo-v3
            detections = write_results(detections, args.confidence, num_classes, device, nms=True, nms_conf=args.nms_thresh)
            
            #### detections = ros_message_rois()
            #### ros_rois --> [0,0, x1, y1, x2, y2] 

            # original image dimension --> im_dim
            
            #view_image(detections)
            text_img = np.zeros((200, 1750, 3))

            if type(detections) != int: 
                if detections.shape[0]:
                    bboxs = detections[:, 1:5].clone()

                    im_dim = im_dim.repeat(detections.shape[0], 1)
                    scaling_factor = torch.min(inp_dim/im_dim, 1)[0].view(-1, 1)
                    
                    detections[:, [1, 3]] -= (inp_dim - scaling_factor*im_dim[:, 0].view(-1, 1))/2
                    detections[:, [2, 4]] -= (inp_dim - scaling_factor*im_dim[:, 1].view(-1, 1))/2

                    detections[:, 1:5] /= scaling_factor

                    small_object_ratio = torch.FloatTensor(detections.shape[0])

                    for i in range(detections.shape[0]):
                        detections[i, [1, 3]] = torch.clamp(detections[i, [1, 3]], 0.0, im_dim[i, 0])
                        detections[i, [2, 4]] = torch.clamp(detections[i, [2, 4]], 0.0, im_dim[i, 1])

                        object_area = (detections[i, 3] - detections[i, 1])*(detections[i, 4] - detections[i, 2])
                        orig_img_area = im_dim[i, 0]*im_dim[i, 1]
                        small_object_ratio[i] = object_area/orig_img_area
                    
                    detections = detections[small_object_ratio > 0.05]
                    im_dim = im_dim[small_object_ratio > 0.05]
                    
                    if detections.size(0) > 0:
                        feature = yolov3.get_feature()
                        feature = feature.repeat(detections.size(0), 1, 1, 1)

                        orig_img_dim = im_dim[:, 1:]
                        orig_img_dim = orig_img_dim.repeat(1, 2)
                    
                        scaling_val = 16 

                        bboxs /= scaling_val
                        bboxs = bboxs.round()
                        bboxs_index = torch.arange(bboxs.size(0), dtype=torch.int)
                        bboxs_index = bboxs_index.to(device)
                        bboxs = bboxs.to(device)

                        roi_align = RoIAlign(args.roi_size, args.roi_size, transform_fpcoor=True).to(device)
                        roi_features = roi_align(feature, bboxs, bboxs_index)

                        outputs = encoder(roi_features)                        
                        
                        for i in range(detections.shape[0]):

                            sampled_caption = []
                            #attr_fc = outputs[]
                            for j in range(len(outputs)):
                                max_index = torch.max(outputs[j][i].data, 0)[1]
                                word = attribute_pool[j][max_index]
                                sampled_caption.append(word)
                            
                            c11 = sampled_caption[11]
                            sampled_caption[11] = sampled_caption[10]
                            sampled_caption[10] = c11

                            sentence = ' '.join(sampled_caption)
                             
                            sys.stdout.write('                                                                                        ' + '\r')

                            sys.stdout.write(sentence + '             '+ '\r')
                            sys.stdout.flush()
                            write(detections[i], orig_img, sentence, i+1, coco_classes, colors)
                            
                            cv2.putText(text_img, sentence, (0, i*40+35), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 1 )

            cv2.imshow("frame", orig_img)
            cv2.imshow("text", text_img)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            if key & 0xFF == ord('w'):
                wait(0)
            if key & 0xFF == ord('s'):
                continue    
            frames += 1
            #print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
            
        else:
            break
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()  
    parser.add_argument('--encoder_path', type=str, default='encoder-12-1170.ckpt', help='path for trained encoder')
   
    # Encoder - Yolo-v3 parameters
    parser.add_argument('--confidence', type=float, default = 0.5, help = 'Object Confidence to filter predictions')
    parser.add_argument('--nms_thresh', type=float , default = 0.4, help = 'NMS Threshhold')
    parser.add_argument('--cfg_file', type = str, default = 'cfg/yolov3.cfg', help ='Config file')
    parser.add_argument('--weights_file', type = str, default = 'yolov3.weights', help = 'weightsfile')
    parser.add_argument('--reso', type=str, default = '416', help = 'Input resolution of the network. Increase to increase accuracy. Decrease to increase speed')
    parser.add_argument('--scales', type=str, default = '1,2,3', help =  'Scales to use for detection')

    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    parser.add_argument('--roi_size', type=int , default=13)
    args = parser.parse_args()
     
    coco_classes = load_classes('data/coco.names')
    colors = pkl.load(open("pallete2", "rb"))
    
    main(args)   
