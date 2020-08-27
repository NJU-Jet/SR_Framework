#python train.py --opt options/train/base.yaml --name baseline
#python train.py --opt options/train/DHF.yaml --name DHF_bs16ps64lr4e-4 --lr 4e-4 --ps 64
python train.py --opt options/train/DHF.yaml --name DHF_bs16ps128lr4e-4_Y --lr 4e-4 --train_Y
