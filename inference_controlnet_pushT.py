import argparse
import logging
import os
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
import clip
from model import (
    UNetwithControl,
    SensorModel,
    ControlNet,
    StatefulControlNet,
    StatefulUNet,
)
from PIL import Image
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
import copy
import time
import random
import pickle
import json
import pdb

import sys
sys.path.append('/home/yuyao/diffusion_policy/diffusion_policy/dataset/')
sys.path.append('/home/yuyao/diffusion_policy/')
from pusht_image_dataset import PushTImageDataset

class PushTInferenceEngine:
    def __init__(self, model_path_1, model_path_2):
        self.batch_size = 1
        self.dim_x = 10
        self.dim_gt = 10
        self.channel_img_1 = 2
        self.win_size = 24
        self.global_step = 0
        self.mode = "Test"
        self.checkpoint_path_1 = model_path_1
        self.checkpoint_path_2 = model_path_2

        dataset = "duck"  # "duck"

        if dataset == "Drum":
            self.base_model = StatefulUNet(dim_x=self.dim_x, window_size=self.win_size)
            self.model = StatefulControlNet(dim_x=self.dim_x, window_size=self.win_size)
        else:
            self.base_model = UNetwithControl(
                dim_x=self.dim_x, window_size=self.win_size
            )
            self.model = ControlNet(dim_x=self.dim_x, window_size=self.win_size)
        self.sensor_model = SensorModel(
            state_est=1,
            dim_x=self.dim_x,
            emd_size=256,
            input_channel=self.channel_img_1,
        )

        self.pushTImageDataset = PushTImageDataset(zarr_path='/home/yuyao/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr', val_ratio=0.2)
        self.normalizer = self.pushTImageDataset.get_normalizer()

        # -----------------------------------------------------------------------------#
        # ----------------------------    diffusion API     ---------------------------#
        # -----------------------------------------------------------------------------#
        num_diffusion_iters = 50
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_diffusion_iters,
            # the choise of beta schedule has big impact on performance
            # we found squared cosine works the best
            beta_schedule="squaredcos_cap_v2",
            # clip output to [-1,1] to improve stability
            clip_sample=True,
            # our network predicts noise (instead of denoised action)
            prediction_type="epsilon",
        )

        # -----------------------------------------------------------------------------#
        # ---------------------------    get model ready     --------------------------#
        # -----------------------------------------------------------------------------#
        # Check model type
        if not isinstance(self.model, nn.Module):
            raise TypeError("model must be an instance of nn.Module")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            self.model.cuda()
            self.sensor_model.cuda()
            self.base_model.cuda()
        self.clip_model, preprocess = clip.load("ViT-B/32", device=self.device)

        # -----------------------------------------------------------------------------#
        # ---------------------------------    setup     ------------------------------#
        # -----------------------------------------------------------------------------#
        if torch.cuda.is_available():
            checkpoint_1 = torch.load(self.checkpoint_path_1)
            self.model.load_state_dict(checkpoint_1["model"])
            checkpoint_2 = torch.load(self.checkpoint_path_2)
            self.sensor_model.load_state_dict(checkpoint_2["model"])
        else:
            checkpoint_1 = torch.load(
                self.checkpoint_path_1, map_location=torch.device("cpu")
            )
            checkpoint_2 = torch.load(
                self.checkpoint_path_2, map_location=torch.device("cpu")
            )
            self.model.load_state_dict(checkpoint_1["model"])
            self.sensor_model.load_state_dict(checkpoint_2["model"])

        # -----------------------------------------------------------------------------#
        # ---------------------------   create base model   ---------------------------#
        # -----------------------------------------------------------------------------#
        self.base_model.time_mlp.load_state_dict(self.model.time_mlp.state_dict())
        self.base_model.lang_model.load_state_dict(self.model.lang_model.state_dict())
        self.base_model.fusion_layer.load_state_dict(
            self.model.fusion_layer.state_dict()
        )
        self.base_model.downs.load_state_dict(self.model.downs.state_dict())
        self.base_model.mid_block1.load_state_dict(self.model.mid_block1.state_dict())
        self.base_model.mid_block2.load_state_dict(self.model.mid_block2.state_dict())
        self.base_model.ups.load_state_dict(self.model.ups.state_dict())
        self.base_model.final_conv.load_state_dict(self.model.final_conv.state_dict())

        if dataset == "Drum":
            self.base_model.addition_module.load_state_dict(
                self.model.addition_module.state_dict()
            )

        self.base_model.eval()
        self.model.eval()
        self.sensor_model.eval()

    def base_policy(self, img, sentence='pick yellow duck'):
        # -----------------------------------------------------------------------------#
        # ---------------------------------    test      ------------------------------#
        # -----------------------------------------------------------------------------#
        # get data ready - sentence
        sentence = clip.tokenize([sentence]).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(sentence)
            text_features = text_features.clone().detach()
            text_features = text_features.to(torch.float32)
            text_features = text_features.repeat(32, 1)

        # -----------------------------------------------------------------------------#
        # ------------------------------ diffusion sampling ---------------------------#
        # -----------------------------------------------------------------------------#
        self.noise_scheduler.set_timesteps(50)
        # initialize action from Guassian noise
        noisy_action = torch.randn((32, self.dim_x, self.win_size)).to(self.device)
        with torch.no_grad():
            img_emb = self.sensor_model(img)
            for k in self.noise_scheduler.timesteps:
                # predict noise
                t = torch.stack([k]).repeat(32).to(self.device)
                predicted_noise = self.base_model(
                    noisy_action, img_emb, text_features, t
                )

                # inverse diffusion step (remove noise)
                noisy_action = self.noise_scheduler.step(
                    model_output=predicted_noise, timestep=k, sample=noisy_action
                ).prev_sample
            pred = self.normalizer['action'].unnormalize(noisy_action[:,:2,:].detach())
            raw_pred = noisy_action.detach()
            
        return pred, raw_pred

    def controlnet(self, img, raw_pred):
        # -----------------------------------------------------------------------------#
        # ---------------------------------    test      ------------------------------#
        # -----------------------------------------------------------------------------#
        # get data ready - sentence
        sentence = 'pick yellow duck'
        sentence = clip.tokenize([sentence]).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(sentence)
            text_features = text_features.clone().detach()
            text_features = text_features.to(torch.float32)
            text_features = text_features.repeat(32, 1)

        # -----------------------------------------------------------------------------#
        # ------------------------------ diffusion sampling ---------------------------#
        # -----------------------------------------------------------------------------#
        self.noise_scheduler.set_timesteps(50)
        # initialize action from Guassian noise
        noisy_action = torch.randn((32, self.dim_x, self.win_size)).to(self.device)
        with torch.no_grad():
            img_emb = self.sensor_model(img)
            for k in self.noise_scheduler.timesteps:
                # predict noise
                t = torch.stack([k]).repeat(32).to(self.device)
                predicted_noise = self.model(
                    noisy_action, img_emb, text_features, raw_pred, t
                )
                
                # inverse diffusion step (remove noise)
                noisy_action = self.noise_scheduler.step(
                    model_output=predicted_noise, timestep=k, sample=noisy_action
                ).prev_sample
            pred = self.normalizer['action'].unnormalize(noisy_action[:,:2,:].detach())
            raw_pred = noisy_action.detach()
        return pred, raw_pred


if __name__ == "__main__":
    # gat path to the model
    path_1 = "path_to_controlnet/controlnet-model-153000"
    path_2 = "path_to_sensor_model/sensor_model-225600"
    inference = Engine(path_1, path_2)

    # run inference
    img = Image.open("0.jpg")
    sentence = "pick up the "
    # sentence = 0

    # run diffusion
    save = []
    for i in range(1):
        pred, raw_pred = inference.base_policy(img, sentence)
        save.append(pred)
    with open("diffusion_out", "wb") as f:
        pickle.dump(save, f)

    # run controlnet
    with open(os.path.join("outputs.json"), "r") as fh:
        traj = json.load(fh)["traj"]
        traj = np.array(traj)
    traj = transform(traj)
    traj = torch.tensor(traj, dtype=torch.float32)
    traj = rearrange(traj, "(n dim) time -> n dim time", n=1)
    raw_pred = traj
    save = []
    for i in range(1):
        pred, raw_pred = inference.controlnet(img, sentence, raw_pred)
        save.append(pred)
    with open("controlnet_out", "wb") as f:
        pickle.dump(save, f)
