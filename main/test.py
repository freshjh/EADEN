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




cfg.domains = ['ADM', 'BigGAN', 'glide', 'Midjourney', 'SDV15', 'SDV14', 'VQDM', 'wukong']

assert cfg.ckpt_path, "Please specify the path to the model checkpoint"
model_name = os.path.basename(cfg.ckpt_path).replace(".pth", "")
rows = []
print(f"'{cfg.exp_name}:{model_name}' model testing on...")

avg = numpy.zeros(4)


for i, domain in enumerate(cfg.domains):
    cfg.domain = domain
    
    model = get_network(cfg.arch)
    state_dict = torch.load(cfg.ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict["model"])
    model.cuda()
    model.eval()

    test_results = validate(model, cfg)
    print(f"{cfg.datasets[0]}->{cfg.domain}:")

    
    for k, v in test_results.items():
        
        print(f"{k}: {v:.5f}", end=" ")
        
    
    print()
    print("*" * 50)
    if i == 0:
        rows.append(["TestSet"] + list(test_results.keys()))
    rows.append([cfg.domain] + list(test_results.values()))

    avg += numpy.array(list(test_results.values()))

avg=  avg/len(cfg.domains)

avg = numpy.round(avg,4)


rows.append(['avg']+avg.tolist())

results_dir = os.path.join(cfg.root_dir, "data", "results", cfg.exp_name)
os.makedirs(results_dir, exist_ok=True)
with open(os.path.join(results_dir, f"{model_name}.csv"), "w") as f:
    csv_writer = csv.writer(f, delimiter=",")
    csv_writer.writerows(rows)
