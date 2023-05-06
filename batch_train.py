import os

modarity = 'a'
model_dir = './data/chenxiahan/result/lymph-node-metastasis-classification/version6/seed/'
data_dir_rotate_a = './data/ruijin_Lymph_node/data_rotate/data_rotate_a_40/'
data_dir_rotate_v = './data/ruijin_Lymph_node/data_rotate/data_rotate_v_40/'
data_dir_projection_a = './data/ruijin_Lymph_node/Mercator_Projection/Mercator_Projection2a/'
data_dir_projection_v = './data/ruijin_Lymph_node/Mercator_Projection/Mercator_Projection2v/'
data_image_list = './label/label.txt'

# size = 64
seed = [1111]
model = 'resnet18'
cuda = 0
# Baseline + Aug + self-supervised
time = 3
for seed_ in seed:
    path = '{}_143_157x{}_b64_lr0.0001_leaky0.1_p1wdfc0.01_aug2_epoch20_80_rotate_projection_seed{}_pre_augshuffle_dp0.5'.format(model, time, seed_)
    for fold_index_ in range(0, 5):
        print(path, fold_index_)
        os.system("CUDA_VISIBLE_DEVICES={cuda} python train.py "
                  "--model {} "
                  "--pretrained True "
                  "--n_epoch 20 "
                  "--epochs 80 "
                  "--num_classes 2 "
                  "--batch-size 64 "
                  "--lr 0.0001 "
                  "--seed {seed} "
                  "--fold {fold_index} "
                  "--times {time} "
                  "--weight-decay-fc 0.01 "
                  "--weight-decay-ssl 2 "
                  "--model-dir {model_dir} "
                  "--data-dir-rotate-a {data_dir_rotate_a} "
                  "--data-dir-rotate-v {data_dir_rotate_v} "
                  "--data-dir-projection-a {data_dir_projection_a} "
                  "--data-dir-projection-v {data_dir_projection_v} "
                  "--data-image-list {data_image_list} "
                  "--resume {model_dir}{path}/checkpoint/ "
                  "--name {path}".format(model, seed=seed_, fold_index=fold_index_, path=path, time=time,
                                         model_dir=model_dir, data_image_list=data_image_list, cuda=cuda,
                                         data_dir_rotate_a=data_dir_rotate_a, data_dir_rotate_v=data_dir_rotate_v,
                                         data_dir_projection_a=data_dir_projection_a, data_dir_projection_v=data_dir_projection_v))
    for fold_index_ in range(0, 5):
        os.system("CUDA_VISIBLE_DEVICES={cuda} python test.py "
                  "--model {} "
                  "--seed {seed} "
                  "--num_classes 2 "
                  "--batch-size 64 "
                  "--fold {fold_index} "
                  "--times {time} "
                  "--model-dir {model_dir} "
                  "--data-dir-rotate-a {data_dir_rotate_a} "
                  "--data-dir-rotate-v {data_dir_rotate_v} "
                  "--data-dir-projection-a {data_dir_projection_a} "
                  "--data-dir-projection-v {data_dir_projection_v} "
                  "--data-image-list {data_image_list} "
                  "--resume {model_dir}{path}/checkpoint/ "
                  "--name {path}".format(model, seed=seed_, fold_index=fold_index_, path=path, time=time,
                                         model_dir=model_dir, data_image_list=data_image_list, cuda=cuda,
                                         data_dir_rotate_a=data_dir_rotate_a, data_dir_rotate_v=data_dir_rotate_v,
                                         data_dir_projection_a=data_dir_projection_a, data_dir_projection_v=data_dir_projection_v))