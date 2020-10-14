
from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2 
import random

from bbox import bbox_iou, bbox_iou2
import sys


def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = cv2.resize(orig_im, (inp_dim, inp_dim))
#    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = img.transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim

def load_image2(image_path, transform=None):
    image = Image.open(image_path)
    orig_img = np.array(image)
    dim = image.size
    image = image.resize([416, 416], Image.LANCZOS)  #224 416
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image, orig_img, dim

def write(x, img, phrases, coco_classes, colors):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    if cls == 0:  # for human bbox only detection
        label = "{0}".format(coco_classes[cls])
        color = random.choice(colors)
        cv2.rectangle(img, c1, c2,color, 3)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2, color, -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
        for i, phrase in enumerate(phrases):
            cv2.putText(img, phrase, (c1[0]-200, c1[1] + t_size[1] + 150 + i*20), cv2.FONT_HERSHEY_PLAIN, 1, color, 1 )
    return img

def write(x, img, raw_pharses,phrases, order, coco_classes, colors):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    attrs_name = ['Tops', 'color', 'pattern', 'gender', 'season', 'type', 'spleeves', 'Bottoms', 'color', 'pattern',
                  'gender', 'season', 'spleeves', 'type', "legpose"]

    if cls == 0:  # for human bbox only detection
        #label = "{0}: {1}".format(coco_classes[cls], order)
        label = "                            "
        color = random.choice(colors)
        cv2.rectangle(img, c1, c2,color, 3)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + len(attrs_name)*t_size[1] + 4
        cv2.rectangle(img, c1, c2, color, -1)

        j = 0
        for i in range(len(attrs_name)):
            if attrs_name[i] == "Bottoms" or attrs_name[i] == "Tops":
                text = attrs_name[i]
                j = j + 1
            else:

                text = "  "+attrs_name[i] + ":" + raw_pharses[i - j]

            cv2.putText(img, label + " " + text, (c1[0] -t_size[0], c1[1] + t_size[1] + 5 + i*t_size[1]), cv2.FONT_HERSHEY_PLAIN, 1,
                        [225, 255, 255], 1)


        #cv2.putText(img, label+" "+phrases, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)

        #for i, phrase in enumerate(phrases):
        #    cv2.putText(img, phrase, (c1[0], c1[1] + t_size[1] + 150 + i*20), cv2.FONT_HERSHEY_PLAIN, 1, color, 1 )
    return img

def view_image(bboxes):
    img = np.full((416, 416, 3), 100, dtype='uint8')
    for bbox in bboxes:
        c1 = tuple(bbox[1:3].int())
        c2 = tuple(bbox[3:5].int())
        cv2.rectangle(img, c1, c2, 128+20, 3)
    cv2.imshow("01", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def sampling_sentence(samples, probs, vocab):
    
    j = 0
    captions = []
    for sampled_ids in samples:
        sampled_ids = sampled_ids[0].cpu().numpy()  
                # (1, max_seq_length) -> (max_seq_length)
        probs_ids = probs[j][0].detach()
        probs_ids = probs_ids.cpu().numpy() 
        if j == 1 :
            break 
        j+=1
        
        # Convert word_ids to words
        sampled_caption = list()
        t_caption = list()
        no_flag = False
        for word_id in sampled_ids:
            word = vocab.idx2word[word_id]
            if word == '<end>':
                break
            sampled_caption.append(word)
            
        for word in sampled_caption:
            if word == 'no':
                no_flag = True
                continue
            else:
                if not no_flag:
                    t_caption.append(word)
                    no_flag = False
                else:
                    continue
        if t_caption.__len__() > 0:
            if t_caption[-1] == 'and':
                t_caption.pop()
        sentence = ' '.join(t_caption)
        p = sentence.find('and')
        if p > 0:
            upper = sentence[:p]
            if upper.find('vest') > -1:
                upper = upper + ' no sleeves'
            captions.append(upper)
            captions.append(sentence[p:])
        else:
            if sentence.find('vest') > -1:
                sentence = sentence + ' no sleeves'
            captions.append(sentence)
        #sentence = ' '.join(sampled_caption)
        
        # Print out the image and the generated caption
        #print (sentence)
        #sys.stdout.write(sentence)
        #print (probs_ids)
        #captions.append(sentence)
    return captions

def sampling_sentence1(sampled_ids, vocab):
    sampled_ids = sampled_ids[0].cpu().numpy()  
            # (1, max_seq_length) -> (max_seq_length)
   
    # Convert word_ids to words
    sampled_caption = list()
    t_caption = list()
    no_flag = False
    for word_id in sampled_ids[0]:
        word = vocab.idx2word[word_id]
        if word == '<end>':
            break
        sampled_caption.append(word)
        
    for word in sampled_caption:
        if word == 'no':
            no_flag = True
            continue
        else:
            if not no_flag:
                t_caption.append(word)
                no_flag = False
            else:
                continue
    if t_caption.__len__() > 0:
        if t_caption[-1] == 'and':
            t_caption.pop()
    sentence = ' '.join(t_caption)
   
    
    # Print out the image and the generated caption
    print (sentence)
    #print (probs_ids)
    
    
    return sentence

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def count_learnable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def convert2cpu(matrix):
    if matrix.is_cuda:
        return torch.FloatTensor(matrix.size()).copy_(matrix)
    else:
        return matrix

def predict_transform(prediction, inp_dim, anchors, num_classes, device):
    batch_size = prediction.size(0)
    stride =  inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)
    
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)


    #Sigmoid the  centre_X, centre_Y. and object confidencce
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])
    

    #Add the center offsets
    grid_len = np.arange(grid_size)
    a,b = np.meshgrid(grid_len, grid_len)
    
    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)
    

    x_offset = x_offset.to(device)
    y_offset = y_offset.to(device)
    
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)
    
    prediction[:,:,:2] += x_y_offset
      
    #log space transform height and the width
    anchors = torch.FloatTensor(anchors)
    

    anchors = anchors.to(device)
    
    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors

    #Softmax the class scores
    prediction[:,:,5: 5 + num_classes] = torch.sigmoid((prediction[:,:, 5 : 5 + num_classes]))

    prediction[:,:,:4] *= stride
   
    
    return prediction

def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names

def get_im_dim(im):
    im = cv2.imread(im)
    w,h = im.shape[1], im.shape[0]
    return w,h

def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)
    
    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res

def write_results(prediction, confidence, num_classes, device, nms = True, nms_conf = 0.4):
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    prediction = prediction*conf_mask
    

    try:
        ind_nz = torch.nonzero(prediction[:,:,4]).transpose(0,1).contiguous()
    except:
        return 0
    
    
    box_a = prediction.new(prediction.shape)
    box_a[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_a[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_a[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) 
    box_a[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_a[:,:,:4]
    

    
    batch_size = prediction.size(0)
    
    output = prediction.new(1, prediction.size(2) + 1)
    write = False


    for ind in range(batch_size):
        #select the image from the batch
        image_pred = prediction[ind]
        

        
        #Get the class having maximum score, and the index of that class
        #Get rid of num_classes softmax scores 
        #Add the class index and the class score of class having maximum score
        max_conf, max_conf_score = torch.max(image_pred[:,5:5+ num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:,:5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)
        

        
        #Get rid of the zero entries
        non_zero_ind =  (torch.nonzero(image_pred[:,4]))

        
        image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7)
        
        #Get the various classes detected in the image
        try:
            img_classes = unique(image_pred_[:,-1])
        except:
             continue
        #WE will do NMS classwise
        for cls in img_classes:
            if cls == 0:
                #get the detections with one particular class
                cls_mask = image_pred_*(image_pred_[:,-1] == cls).float().unsqueeze(1)
                class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()
                

                image_pred_class = image_pred_[class_mask_ind].view(-1,7)

            
            
                #sort the detections such that the entry with the maximum objectness
                #confidence is at the top
                conf_sort_index = torch.sort(image_pred_class[:,4], descending = True )[1]
                image_pred_class = image_pred_class[conf_sort_index]
                idx = image_pred_class.size(0)
                
                #if nms has to be done
                if nms:
                    #For each detection
                    for i in range(idx):
                        #Get the IOUs of all boxes that come after the one we are looking at 
                        #in the loop
                        try:
                            ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:], device)
                        except ValueError:
                            break
            
                        except IndexError:
                            break
                        
                        #Zero out all the detections that have IoU > treshhold
                        iou_mask = (ious < nms_conf).float().unsqueeze(1)
                        image_pred_class[i+1:] *= iou_mask       
                        
                        #Remove the non-zero entries
                        non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
                        image_pred_class = image_pred_class[non_zero_ind].view(-1,7)
                    
                        

                #Concatenate the batch_id of the image to the detection
                #this helps us identify which image does the detection correspond to 
                #We use a linear straucture to hold ALL the detections from the batch
                #the batch_dim is flattened
                #batch is identified by extra batch column
                
                
                batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
                seq = batch_ind, image_pred_class
                if not write:
                    output = torch.cat(seq,1)
                    write = True
                else:
                    out = torch.cat(seq,1)
                    output = torch.cat((output,out))
    
    return output

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 00:12:16 2018

@author: ayooshmac
"""

def predict_transform_half(prediction, inp_dim, anchors, num_classes, CUDA = True):
    batch_size = prediction.size(0)
    stride =  inp_dim // prediction.size(2)

    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)
    grid_size = inp_dim // stride

    
    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)
    
    
    #Sigmoid the  centre_X, centre_Y. and object confidencce
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])

    
    #Add the center offsets
    grid_len = np.arange(grid_size)
    a,b = np.meshgrid(grid_len, grid_len)
    
    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)
    
    if CUDA:
        x_offset = x_offset.cuda().half()
        y_offset = y_offset.cuda().half()
    
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)
    
    prediction[:,:,:2] += x_y_offset
      
    #log space transform height and the width
    anchors = torch.HalfTensor(anchors)
    
    if CUDA:
        anchors = anchors.cuda()
    
    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors

    #Softmax the class scores
    prediction[:,:,5: 5 + num_classes] = nn.Softmax(-1)(Variable(prediction[:,:, 5 : 5 + num_classes])).data

    prediction[:,:,:4] *= stride
    
    
    return prediction


def write_results_half(prediction, confidence, num_classes, nms = True, nms_conf = 0.4):
    conf_mask = (prediction[:,:,4] > confidence).half().unsqueeze(2)
    prediction = prediction*conf_mask
    
    try:
        ind_nz = torch.nonzero(prediction[:,:,4]).transpose(0,1).contiguous()
    except:
        return 0
    
    
    
    box_a = prediction.new(prediction.shape)
    box_a[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_a[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_a[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) 
    box_a[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_a[:,:,:4]
    
    
    
    batch_size = prediction.size(0)
    
    output = prediction.new(1, prediction.size(2) + 1)
    write = False
    
    for ind in range(batch_size):
        #select the image from the batch
        image_pred = prediction[ind]

        
        #Get the class having maximum score, and the index of that class
        #Get rid of num_classes softmax scores 
        #Add the class index and the class score of class having maximum score
        max_conf, max_conf_score = torch.max(image_pred[:,5:5+ num_classes], 1)
        max_conf = max_conf.half().unsqueeze(1)
        max_conf_score = max_conf_score.half().unsqueeze(1)
        seq = (image_pred[:,:5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)
        
        
        #Get rid of the zero entries
        non_zero_ind =  (torch.nonzero(image_pred[:,4]))
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(),:]
        except:
            continue
        
        #Get the various classes detected in the image
        img_classes = unique(image_pred_[:,-1].long()).half()
        
        
        
                
        #WE will do NMS classwise
        for cls in img_classes:
            #get the detections with one particular class
            cls_mask = image_pred_*(image_pred_[:,-1] == cls).half().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()
            

            image_pred_class = image_pred_[class_mask_ind]

        
             #sort the detections such that the entry with the maximum objectness
             #confidence is at the top
            conf_sort_index = torch.sort(image_pred_class[:,4], descending = True )[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)
            
            #if nms has to be done
            if nms:
                #For each detection
                for i in range(idx):
                    #Get the IOUs of all boxes that come after the one we are looking at 
                    #in the loop
                    try:
                        ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:], device)
                    except ValueError:
                        break
        
                    except IndexError:
                        break
                    
                    #Zero out all the detections that have IoU > treshhold
                    iou_mask = (ious < nms_conf).half().unsqueeze(1)
                    image_pred_class[i+1:] *= iou_mask       
                    
                    #Remove the non-zero entries
                    non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
                    image_pred_class = image_pred_class[non_zero_ind]
                    
                    
            
            #Concatenate the batch_id of the image to the detection
            #this helps us identify which image does the detection correspond to 
            #We use a linear straucture to hold ALL the detections from the batch
            #the batch_dim is flattened
            #batch is identified by extra batch column
            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
            seq = batch_ind, image_pred_class
            
            if not write:
                output = torch.cat(seq,1)
                write = True
            else:
                out = torch.cat(seq,1)
                output = torch.cat((output,out))
    
    return output
