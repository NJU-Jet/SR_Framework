for train:
    (1)modify demo.sh, channge --opt and --name which determines save path;
    (2)create options/train/{model}.yaml if not exists;
    (3)modify options/train/{model}.yaml. Specifically:
        - gpu_ids
        - scale
        - patch_size
        - which model
        - learning rate
        - pretrained
    (4)run sh demo.sh, happily training

for test:
    (1)create options/test/{model}.yaml if not exists;
    (2)modify options/test/{model}.yaml. Specifically:
        - gpu_ids
        - scale
        - mode
        - dataroot_HR
        - dataroot_LR
        - which model
    (3)run sh test_demo.sh, happily testing
