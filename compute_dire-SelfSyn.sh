## set MODEL_PATH, num_samples, has_subfolder, images_dir, recons_dir, dire_dir
export CUDA_VISIBLE_DEVICES=0
# export NCCL_P2P_DISABLE=1
MODEL_PATH="./256x256_diffusion_uncond.pt" # "models/lsun_bedroom.pt, models/256x256_diffusion_uncond.pt"

MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 500 --dropout 0.1 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"

domains=(
    /opt/data/private/Datasets/GenImage/ADM/val/nature
    /opt/data/private/Datasets/GenImage/ADM/val/ai
    /opt/data/private/Datasets/GenImage/BigGAN/val/nature
    /opt/data/private/Datasets/GenImage/BigGAN/val/ai
    /opt/data/private/Datasets/GenImage/Midjourney/val/nature
    /opt/data/private/Datasets/GenImage/Midjourney/val/ai
    /opt/data/private/Datasets/GenImage/SDV14/imagenet_ai_0419_sdv4/val/nature
    /opt/data/private/Datasets/GenImage/SDV14/imagenet_ai_0419_sdv4/val/ai
    /opt/data/private/Datasets/GenImage/SDV15/val/nature
    /opt/data/private/Datasets/GenImage/SDV15/val/ai
    /opt/data/private/Datasets/GenImage/VQDM/val/nature
    /opt/data/private/Datasets/GenImage/VQDM/val/ai
    /opt/data/private/Datasets/GenImage/glide/val/nature
    /opt/data/private/Datasets/GenImage/glide/val/ai
    /opt/data/private/Datasets/GenImage/wukong/val/nature
    /opt/data/private/Datasets/GenImage/wukong/val/ai
)
dire_dirs=(
    /opt/data/private/Datasets/GenImage/ADM/dire500/nature
    /opt/data/private/Datasets/GenImage/ADM/dire500/ai
    /opt/data/private/Datasets/GenImage/BigGAN/dire500/nature
    /opt/data/private/Datasets/GenImage/BigGAN/dire500/ai
    /opt/data/private/Datasets/GenImage/Midjourney/dire500/nature
    /opt/data/private/Datasets/GenImage/Midjourney/dire500/ai
    /opt/data/private/Datasets/GenImage/SDV14/dire500/nature
    /opt/data/private/Datasets/GenImage/SDV14/dire500/ai
    /opt/data/private/Datasets/GenImage/SDV15/dire500/nature
    /opt/data/private/Datasets/GenImage/SDV15/dire500/ai
    /opt/data/private/Datasets/GenImage/VQDM/dire500/nature
    /opt/data/private/Datasets/GenImage/VQDM/dire500/ai
    /opt/data/private/Datasets/GenImage/glide/dire500/nature
    /opt/data/private/Datasets/GenImage/glide/dire500/ai
    /opt/data/private/Datasets/GenImage/wukong/dire500/nature
    /opt/data/private/Datasets/GenImage/wukong/dire500/ai
)
for i in "${!domains[@]}"; do
    domain="${domains[$i]}"
    echo "$domain"
    # 递归查询domain下有多少文件
    num_samples=$(find $domain -type f | wc -l)
    echo $num_samples
    SAMPLE_FLAGS="--batch_size 16 --num_samples $num_samples --timestep_respacing ddim20 --use_ddim True"
    SAVE_FLAGS="--images_dir $domain --recons_dir ${dire_dirs[$i]} --dire_dir ${dire_dirs[$i]}"
    # echo $SAVE_FLAGS
    # mpiexec -n 2 python compute_dire.py --model_path $MODEL_PATH $MODEL_FLAGS  $SAVE_FLAGS $SAMPLE_FLAGS --has_subfolder True
    python compute_dire-SelfSyn.py --model_path $MODEL_PATH $MODEL_FLAGS  $SAVE_FLAGS $SAMPLE_FLAGS --has_subfolder False
done