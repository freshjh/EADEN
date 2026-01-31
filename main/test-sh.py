import os
import subprocess



def test_exp(EXP_NAME, GPU):
    # ckpt目录
    EXP_DIR = os.path.abspath("./data/exp/{}/ckpt".format(EXP_NAME))
    CSV_DIR = os.path.abspath("./data/results/{}".format(EXP_NAME))

    # 检查目录是否存在
    if not os.path.exists(EXP_DIR):
        print("Directory does not exist: {}".format(EXP_DIR))
        exit(1)
        
    if not os.path.exists(CSV_DIR):
        # 创建目录
        os.makedirs(CSV_DIR)
        ignore_list = []
    else:
        ignore_list = sorted([d for d in os.listdir(CSV_DIR) if d.endswith('.csv')])


    # 获取所有子目录，并按字母顺序排序
    dir_names = sorted([d for d in os.listdir(EXP_DIR) if d.endswith('.pth')])


    for existed in ignore_list:
        model_name = existed.split(".")[0]
        print("Ignore: {}".format(model_name))
        dir_names.remove(model_name+'.pth')



    # 遍历所有子目录
    for dir_name in dir_names:

        CKPT = os.path.join(EXP_DIR, dir_name)
        
        # 构造命, 令并执行
        command = [
            "python", "test-part.py",
            "--gpus", GPU,
            "--ckpt", CKPT,
            "--exp_name", EXP_NAME,
            "batch_size", "128",
            "datasets_test", "GenImage"
        ]
        subprocess.run(command)




    rows = ['epoch,Acc,AP,R_ACC,F_ACC\n']

    for d in os.listdir(CSV_DIR):
        if d.endswith('.csv'):
            indx = d.split('.')[0].split('_')[-1]
            
            csv_path = os.path.join(CSV_DIR, d)
            
            # 读取csv文件最后一行第二项
            with open(csv_path, 'r') as f:
                rows.append(f.readlines()[-1].replace('avg', indx))


            
    with open(os.path.join(CSV_DIR, "All.txt"), "a") as f:

        f.writelines(rows)





EXP_NAMES = ["L1"]
GPU = '0'

for EXP_NAME in EXP_NAMES:
    test_exp(EXP_NAME, GPU)

