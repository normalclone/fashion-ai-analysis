import torch
import numpy as np 
import argparse
import pickle 
import os
from os import listdir, getcwd
import os.path as osp
import glob
from torchvision import transforms 
from model import EncoderClothing, DecoderClothing
from darknet import Darknet
from PIL import Image
from util import *
import cv2
import pickle as pkl
import random
from preprocess import prep_image


import sys
if sys.version_info >= (3,0):
    from roi_align.roi_align import RoIAlign
else :
    from roi_align import RoIAlign

#  "winter scarf", "cane", "bag", "shoes", "hat", "face"]
#attribute categories = #6
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

attribute_pool = [colors_a, pattern_a, gender_a, season_a, upper_t_a, u_sleeves_a,
                  colors_a, pattern_a, gender_a, season_a, lower_t_a, l_sleeves_a, leg_pose_a]

# Device configuration
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
def main(args):
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor()])
    
    # Load vocabulary wrapper

    # Build the models
    #CUDA = torch.cuda.is_available()

    num_classes = 80
    yolov3 = Darknet(args.cfg_file)
    yolov3.load_weights(args.weights_file)


    print("yolo-v3 network successfully loaded")

    attribute_size = [15, 7, 3, 5, 8, 4, 15, 7, 3, 5, 3, 3, 4]

    encoder = EncoderClothing(args.embed_size, device, args.roi_size, attribute_size)

    # Prepare an image   
    images = "test"

    
    try:
        list_dir = os.listdir(images)
     #   list_dir.sort(key=lambda x: int(x[:-4]))
        imlist = [osp.join(osp.realpath('.'), images, img) for img in list_dir if os.path.splitext(img)[1].lower() =='.jpg'  or os.path.splitext(img)[1].lower() =='.jpeg' or os.path.splitext(img)[1] =='.png']
        print(imlist)
    except NotADirectoryError:
        imlist = []
        imlist.append(osp.join(osp.realpath('.'), images))
        print('Not a directory error')
    except FileNotFoundError:
        print ("No file or directory with the name {}".format(images))
        exit()

    yolov3.to(device)
    encoder.to(device)
    
    yolov3.eval()
    encoder.eval()

    encoder.load_state_dict(torch.load(args.encoder_path, map_location=device))

    for inx, image_filename in enumerate(imlist):
        print ('---------------------------')
        print(image_filename)
        preload_img = cv2.imread(image_filename)
        pixel = preload_img.shape[1] * preload_img.shape[0]

        if pixel < 352 * 240:
            yolov3.net_info["height"] = 1024
        if pixel >= 352 * 240:
            yolov3.net_info["height"] = 1024
        if pixel >= 480 * 360:
            yolov3.net_info["height"] = 1024
        if pixel >= 858 * 480:
            yolov3.net_info["height"] = 832
        if pixel >= 1280 * 720:
            yolov3.net_info["height"] = 832

        inp_dim = int(yolov3.net_info["height"])

        image, orig_img, im_dim = prep_image(image_filename, inp_dim)

        im_dim = torch.FloatTensor(im_dim).repeat(1, 2)

        image_tensor = image.to(device)
        im_dim = im_dim.to(device)
        
        # Generate an caption from the image
        detections = yolov3(image_tensor, device, True) # prediction mode for yolo-v3
        detections = write_results(detections, args.confidence, num_classes, device, nms=True, nms_conf=args.nms_thresh)
        # original image dimension --> im_dim
        #view_image(detections)


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

                detections = detections[small_object_ratio > 0.02]
                im_dim = im_dim[small_object_ratio > 0.02] 
                
                if detections.size(0) > 0:
                    feature = yolov3.get_feature()
                    feature = feature.repeat(detections.size(0), 1, 1, 1)

                    #orig_img_dim = im_dim[:, 1:]
                    #orig_img_dim = orig_img_dim.repeat(1, 2)
                   
                    scaling_val = 16

                    bboxs /= scaling_val
                    bboxs = bboxs.round()
                    bboxs_index = torch.arange(bboxs.size(0), dtype=torch.int)
                    bboxs_index = bboxs_index.to(device)
                    bboxs = bboxs.to(device)

                    roi_align = RoIAlign(args.roi_size, args.roi_size, transform_fpcoor=True).to(device)
                    roi_features = roi_align(feature, bboxs, bboxs_index)
                #    print(roi_features)
                #    print(roi_features.size())

                    #roi_features = roi_features.reshape(roi_features.size(0), -1)

                    #roi_align_feature = encoder(roi_features)
                    
                    outputs = encoder(roi_features)
                        #attribute_size = [15, 7, 3, 5, 7, 4, 15, 7, 3, 5, 4, 3, 4]
                    #losses = [criteria[i](outputs[i], targets[i]) for i in range(len(attribute_size))]

                    for i in range(detections.shape[0]):

                        sampled_caption = []
                        #attr_fc = outputs[]
                        for j in range(len(outputs)):
                            #temp = outputs[j][i].data
                            max_index = torch.max(outputs[j][i].data, 0)[1]
                            word = attribute_pool[j][max_index]
                            sampled_caption.append(word)
                        
                        c11 = sampled_caption[11]
                        sampled_caption[11] = sampled_caption[10]
                        sampled_caption[10] = c11

                        sentence = '/'.join(sampled_caption)
                 
                        # again sampling for testing
                        
                        print (str(i+1) + ': ' + sentence)
                        write(detections[i], orig_img, sampled_caption,sentence, i+1, coco_classes, colors)
                        #list(map(lambda x: write(x, orig_img, captions), detections[i].unsqueeze(0)))
        
        cv2.imshow("frame", orig_img)
        key = cv2.waitKey(0)

        if key & 0xFF == ord('q'): 
            break

#    image = Image.open(args.image)   
#    plt.imshow(np.asarray(image))   
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()  
    
    parser.add_argument('--encoder_path', type=str, default='encoder-12-1170.ckpt', help='path for trained encoder')
   
    parser.add_argument('--vocab_path1', type=str, default='json/train_up_vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--vocab_path2', type=str, default='clothing_vocab_accessory2.pkl', help='path for vocabulary wrapper')
   
    # Encoder - Yolo-v3 parameters 
    parser.add_argument('--confidence', type=float, default = 0.5, help = 'Object Confidence to filter predictions')
    parser.add_argument('--nms_thresh', type=float , default = 0.4, help = 'NMS Threshhold')
    parser.add_argument('--cfg_file', type = str, default = 'cfg/yolov3.cfg', help ='Config file')
    parser.add_argument('--weights_file', type = str, default = 'yolov3.weights', help = 'weightsfile')
    parser.add_argument('--reso', type=str, default = '832', help = 'Input resolution of the network. Increase to increase accuracy. Decrease to increase speed')
    parser.add_argument('--scales', type=str, default = '1,2,3', help =  'Scales to use for detection')

    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    parser.add_argument('--roi_size', type=int, default = 13)
    args = parser.parse_args()
     
    coco_classes = load_classes('data/coco.names')
    colors = pkl.load(open("pallete2", "rb"))
    
    main(args)


    
