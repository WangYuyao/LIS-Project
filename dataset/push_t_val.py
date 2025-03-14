import torch
import torch.nn.functional as F

# from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import os
import numpy as np
from skimage import io
from matplotlib import pyplot as plt
import json
import bisect
import clip
import random
from PIL import Image
from einops import rearrange, repeat
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
import pdb

import sys
sys.path.append('/home/yuyao/diffusion_policy/diffusion_policy/dataset/')
sys.path.append('/home/yuyao/diffusion_policy/')
from pusht_image_dataset import PushTImageDataset
import zarr
print(sys.path)

class PushT_val(Dataset):
    def __init__(self, data_dirs, random=False, length_total=24, image_size=(96, 96)):
       
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_inst_to_verb = {
            "pick": ["pick", "pick up", "raise", "hold"],
        }

        self.pushTImageDataset = PushTImageDataset(zarr_path=data_dirs, val_ratio=0.2).get_validation_dataset()
        normalizer = self.pushTImageDataset.get_normalizer()
        self.n_actions = normalizer['action'].normalize(self.pushTImageDataset.replay_buffer['action'])
        self.n_imgs = normalizer['image'].normalize(self.pushTImageDataset.replay_buffer['img'])
            
        self.length_total = length_total
        self.image_size = image_size
        self.lengths_index = zarr.open('/home/yuyao/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr/meta/episode_ends', mode="r")

    def noun_phrase_template(self, target_id):
        self.noun_phrase = {
            0: {
                "name": ["yellow"],
                "object": ["duck"],
            },
        }
        id_name = np.random.randint(len(self.noun_phrase[target_id]["name"]))
        id_object = np.random.randint(len(self.noun_phrase[target_id]["object"]))
        name = self.noun_phrase[target_id]["name"][id_name]
        obj = self.noun_phrase[target_id]["object"][id_object]
        return (name + " " + obj).strip()

    def verb_phrase_template(self, action_inst):
        if action_inst is None:
            action_inst = random.choice(list(self.action_inst_to_verb.keys()))
        action_id = np.random.randint(len(self.action_inst_to_verb[action_inst]))
        verb = self.action_inst_to_verb[action_inst][action_id]
        return verb.strip()

    def sentence_template(self, action_inst=None):
        sentence = ""
        verb = self.verb_phrase_template(action_inst)
        sentence = sentence + verb
        sentence = sentence + " " + self.noun_phrase_template(0)
        return sentence.strip()
    
    def get_action_sequence(self, index):
        transition_history = 12
        trial_idx = bisect.bisect_right(self.lengths_index, index)
        index_end = index + self.length_total
        trial_end = self.lengths_index[trial_idx]

        if index_end <= trial_end:
            action_seq = self.n_actions[index:index_end].T.detach()    
        else:
            action_seq = self.n_actions[index:trial_end].T.detach() 
            last_col_repeated = action_seq[:,-1].unsqueeze(1).repeat(1, index_end - trial_end)
            action_seq = torch.cat((action_seq, last_col_repeated), dim=1)

        action_seq = action_seq.repeat(5, 1)
        # print("action_seq size", action_seq.size())
        # print(self.n_actions[index:index_end])
        return action_seq

    def __len__(self):
        return self.lengths_index[-1]

    def __getitem__(self, index):
        transition_history = 12
        trial_idx = bisect.bisect_right(self.lengths_index, index)
        if trial_idx == 0:
            if index < transition_history:
                index_pre = 0
            else:
                index_pre = index - transition_history
        else:
            step_idx = index - self.lengths_index[trial_idx - 1]
            if step_idx < transition_history:
                index_pre = self.lengths_index[trial_idx - 1]
            else:
                index_pre = index - transition_history

        # for controlnet
        # if step_idx < transition_history:
        #     step_idx_pre = 0
        # else:
        #     step_idx_pre = step_idx - transition_history

        # torch_data = self.pushTImageDataset.__getitem__(trial_idx)   

        # img = torch_data["obs"]["image"][step_idx]
        img = self.n_imgs[index].permute(2, 0, 1).detach()
        img = F.interpolate(img.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
        sentence = self.sentence_template()
        sentence = clip.tokenize([sentence])
        # action = self.get_actions_only(step_idx, trial_idx)
        # prior_action = self.get_actions_only(step_idx_pre, trial_idx)
        # action = torch_data["action"][step_idx]
        # prior_action = torch_data["action"][step_idx_pre]
        # action = self.n_actions[index].detach()
        action = self.get_action_sequence(index)
        # action = action.repeat(5).unsqueeze(1).repeat(1, self.length_total)
        
        # prior_action = self.n_actions[index_pre].detach()
        # prior_action = prior_action.repeat(5).unsqueeze(1).repeat(1, self.length_total)

        prior_action = self.get_action_sequence(index_pre)
        return img, prior_action, action, sentence[0]


def pad_collate_xy_lang(batch):
    (img, prior_action, action, lang) = zip(*batch)
    img = torch.stack(img)
    prior_action = torch.stack(prior_action)
    action = torch.stack(action)
    lang = torch.stack(lang)
    return img, prior_action, action, lang


""" if __name__ == "__main__":
    data_dirs = "/home/yuyao/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr"
    
    dataset = PushT(data_dirs)
    for item in dataset:
        img, prior_action, action, sentence = item
#         input()
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        collate_fn=pad_collate_xy_lang,
    )
#     # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     # clip_model, preprocess = clip.load("ViT-B/32", device=device)

    for item in dataset:
        (images, prior_action, action, sentence) = item
        print(
            images.shape,
        #     prior_state.shape,
            prior_action.shape,
            action.shape,
#         #     pre_action.shape,
            sentence.shape,
#         #     target_pos.shape,
        )
#         print("=======", action.shape)
#         # with torch.no_grad():
#         #     text_features = clip_model.encode_text(sentence)
        input() """

# data = PushT(data_dirs='/home/yuyao/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr')
# print(data[0].shape)
# print(data.get_action_sequence(25640))