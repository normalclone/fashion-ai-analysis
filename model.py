import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
from roi_align.roi_align import RoIAlign
#from darknet import Darknet

import cv2

def view_image(bboxes):
    img = np.full((416, 416, 3), 100, dtype='uint8')
    for bbox in bboxes:
        c1 = tuple(bbox[0:2].int()*16)
        c2 = tuple(bbox[2:4].int()*16)
        cv2.rectangle(img, c1, c2, 128, 3)
    cv2.imshow("02", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

class EncoderClothing(nn.Module):
    def __init__(self, embed_size, device, pool_size, attribute_dim):
        """Load the pretrained yolo-v3 """
        super(EncoderClothing, self).__init__()
        self.device = device
        self.linear = nn.Linear(512*pool_size*pool_size, embed_size)
        self.relu = nn.ReLU()
        #self.module_list = nn.ModuleList([nn.Linear(embed_size, att_size) for att_size in attribute_dim])
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.dropout = nn.Dropout(0.5)
        self.pool_size = pool_size

        self.module_list = nn.ModuleList([self.conv_bn(512, 256, 1, embed_size, att_size) for att_size in attribute_dim])


        
    def conv_bn(self, in_planes, out_planes, kernel_size, embed_size, att_size, stride=1, padding=0, bias=False):
    #"convolution with batchnorm, relu"
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, 1, stride=stride,
                    padding=padding, bias=False),
            nn.BatchNorm2d(out_planes, eps=1e-3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.BatchNorm2d(out_planes, eps=1e-3),
            nn.ReLU(),
            nn.Dropout(0.5),
            Reshape(-1, embed_size),
            nn.Linear(embed_size, att_size)
        )
        
    def forward(self, x):
        """change feature dimension to original image dimension"""
        outputs = {} 
        #features = self.bn(self.linear(features))
        #x = self.relu(self.bn(self.linear(features)))
    #    x = self.dropout(x)
        
        for i in range(len(self.module_list)): 
            output = self.module_list[i](x)
            outputs[i] = output

        return outputs

class EncoderClothing1(nn.Module):
    def __init__(self, embed_size, device, pool_size, attribute_dim):
        """Load the pretrained yolo-v3 """
        super(EncoderClothing, self).__init__()
        self.device = device
        self.linear = nn.Linear(512*pool_size*pool_size, embed_size)
        self.module_list = nn.ModuleList([nn.Linear(512*pool_size*pool_size, att_size) for att_size in attribute_dim])
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.pool_size = pool_size
        
    def forward(self, features):
        """change feature dimension to original image dimension"""
        outputs = {} 
        #features = self.bn(self.linear(features))
        for i in range(len(self.module_list)): 
            x = self.module_list[i](features)
            outputs[i] = x

        return outputs


class DecoderClothing(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, vocab, num_layers, max_seq_length=30):
        """Set the hyper-parameters and build the layers."""
        super(DecoderClothing, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length
        self.vocab = vocab
        self.clothing_class = {
                            'shirt':0, 'jumper':1, 'jacket':2, 'vest':3, 'coat':4,
                            'dress':5, 'pants':6, 'skirt':7, 'scarf':8, 'cane':9, 'bag':10, 'shoes':11,
                            'hat':12, 'face':13, 'glasses':14 }

    def forward(self, features, captions, lengths):   # for training
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions) # [B, 10, 256]   for captions = [B, 10]
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs

    def sample1(self, features, states=None):  # for prediction
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            print(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids
    
    def sample(self, features, states=None):   # for predicton
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        prob_ids = []
        k_samples = []
        k_probs = []
        sampling_num = 30
        prob_thresh = 0.1
        histogram_clothing = np.zeros(15, dtype=int)
        inputs = features.unsqueeze(1)

        for i in range(2):
            hiddens, states = self.lstm(inputs, states)          
            outputs = self.linear(hiddens.squeeze(1))  
            if i == 0 :
                prob_pred, predicted = outputs.max(1)    
                inputs = self.embed(predicted)                       
                inputs = inputs.unsqueeze(1)
            #    states_m = states
            else :
                top_k_prob, top_k = outputs.topk(sampling_num)
                #top_k = top_k.squeeze(0) 
                       
        for i in range(sampling_num):
            inputs = self.embed(top_k[:,i])                      
            inputs = inputs.unsqueeze(1) 
            word_prob = top_k_prob[:,i]
            if word_prob < prob_thresh:
                break
            sampled_ids.append(top_k[:,i])
        #    print(self.vocab.idx2word[top_k[:,i].cpu().numpy()[0]])
            prob_ids.append(word_prob)
            _states = states   # re-load
            duplicate_tag = False

            for j in range(self.max_seg_length):
                _hiddens, _states = self.lstm(inputs, _states)  
                outputs = self.linear(_hiddens.squeeze(1))   
                prob_pred, predicted = outputs.max(1) 
                
                word = self.vocab.idx2word[predicted.cpu().numpy()[0]]
                if word == '<end>':
                    break

                class_index = self.clothing_class.get(word, '')
                if class_index is not '':
                    if histogram_clothing[class_index] > 0:
                        duplicate_tag = True
                        break
                    else:
                        if word == 'jacket' or word == 'coat' or word == 'jumper':
                            class_index = self.clothing_class.get('jacket')
                            histogram_clothing[class_index] += 1
                            class_index = self.clothing_class.get('coat')
                            histogram_clothing[class_index] += 1
                            class_index = self.clothing_class.get('jumper')
                            histogram_clothing[class_index] += 1
                        else:
                            histogram_clothing[class_index] += 1

                sampled_ids.append(predicted)
                prob_ids.append(prob_pred)
                inputs = self.embed(predicted)                      
                inputs = inputs.unsqueeze(1) 
            if duplicate_tag :
                duplicate_tag = False
                sampled_ids = [] 
                prob_ids = []
                continue
            sampled_ids = torch.stack(sampled_ids, 1)  
            prob_ids = torch.stack(prob_ids, 1)                
            k_samples.append(sampled_ids) 
            k_probs.append(prob_ids)
            sampled_ids = [] 
            prob_ids = []

        return k_samples, k_probs
