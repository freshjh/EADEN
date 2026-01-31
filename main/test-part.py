from utils.config import cfg  # isort: split

import csv
import os

import torch, numpy

from utils.eval import get_val_cfg, validate
from utils.utils import get_network

# 忽略掉警告
import warnings
warnings.filterwarnings("ignore")

cfg = get_val_cfg(cfg, split="test", copy=False)



cfg.datasets_test = ['DiTFake', 'aeroblade', 'lsun_bedroom', 'Self-Syhthesis', 'UnivFD', 'GenImage']
cfg.batch_size = 128


cfg.ckpt_path = '/opt/data/private/Projects/AIGCDet/DIRE-Variants/ABLATION-STUDY/Positions/L1/data/exp/L1-GenImage-random-resize/ckpt/model_epoch_7.pth'
assert cfg.ckpt_path, "Please specify the path to the model checkpoint"
model_name = os.path.basename(cfg.ckpt_path).replace(".pth", "")

for dataset in cfg.datasets_test:
    cfg.dataset_test = dataset

    if cfg.dataset_test == 'GenImage':
        cfg.domains = ['BigGAN', 'SDV14', 'ADM', 'glide', 'Midjourney', 'SDV15', 'VQDM', 'wukong']
    elif cfg.dataset_test == 'lsun_bedroom':
        cfg.domains = ['adm', 'dalle2', 'ddpm', 'iddpm', 'if', 'ldm', 'midjourney', 'pndm', 'sdv1', 'sdv2', 'vqdiffusion']
    elif cfg.dataset_test == 'Self-Syhthesis':
        cfg.domains = ['DALLE', 'ddpm', 'guided-diffusion', 'improved-diffusion', 'midjourney']
    elif cfg.dataset_test == 'UnivFD':
        cfg.domains = ['dalle', 'glide_100_10', 'glide_100_27', 'glide_50_27', 'guided', 'ldm_100', 'ldm_200', 'ldm_200_cfg']
    elif cfg.dataset_test == 'aeroblade':
        cfg.domains = ['CompVis-stable-diffusion-v1-1-ViT-L-14-openai', 'kandinsky-community-kandinsky-2-1-ViT-L-14-openai', 'midjourney-v4', 'midjourney-v5', 'midjourney-v5-1', 'runwayml-stable-diffusion-v1-5-ViT-L-14-openai', 'stabilityai-stable-diffusion-2-1-base-ViT-H-14-laion2b_s32b_b79k']
    elif cfg.dataset_test == 'DiTFake':
        cfg.domains = ['FLUX.1-schnell', 'PixArt-Sigma-XL-2-1024-MS', 'stable-diffusion-3-medium-diffusers']

    rows = []

    avg = numpy.zeros(6)


    for i, domain in enumerate(cfg.domains):
        cfg.domain = domain
        
        model = get_network(cfg.arch)
        state_dict = torch.load(cfg.ckpt_path, map_location="cpu")
        model.load_state_dict(state_dict["model"])
        model.cuda()
        model.eval()

        test_results = validate(model, cfg)
        print(f"{cfg.dataset_test}->{cfg.domain}:")

        
        for k, v in test_results.items():
            
            print(f"{k}: {v:.5f}", end=" ")
        print(f"Checking: domain={domain}, v={v}")



        
        print("*" * 50)
        if i == 0:
            rows.append(["TestSet"] + list(test_results.keys()))
        rows.append([cfg.domain] + list(test_results.values()))

        avg += numpy.array(list(test_results.values()))


    avg =  avg/(i+1)

    avg = numpy.round(avg,4)


    rows.append(['avg']+avg.tolist())

    results_dir = os.path.join(cfg.root_dir, "data", "results", 'AUC', 'L1-GenImage-random-resize-fix110-final', cfg.dataset_test)
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, f"{model_name}.csv"), "w") as f:
        csv_writer = csv.writer(f, delimiter=",")
        csv_writer.writerows(rows)