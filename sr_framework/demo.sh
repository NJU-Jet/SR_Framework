#python train.py --opt options/train/base.yaml --name baseline
#python train.py --opt options/train/DHF.yaml --name DHF_bs16ps96lr1e-4 --lr 1e-4 --ps 96 --bs 16 --gpu_ids 1 --use_chop
#python train.py --opt options/train/DHF.yaml --name DHF_bs16ps128lr4e-4_Y --lr 4e-4 --train_Y
#python train.py --opt options/train/IDN.yaml --name IDN_bs64ps64lr1e-4 --lr 1e-4 --bs 64 --ps 64
#python train.py --opt options/train/IDN.yaml --name IDN_bs64ps64lr1e-4_Y --lr 1e-4 --bs 64 --ps 64 --train_Y
#python train.py --opt options/train/EDSR.yaml --name EDSR_bs16ps96lr1e-4  --lr 1e-4 --bs 16 --ps 96 --gpu_ids 2 --use_chop
#python train.py --opt options/train/FTN.yaml --name FTN_bs16ps96lr2e-4  --lr 2e-4 --bs 16 --ps 96 --gpu_ids 1 --use_chop
#python train.py --opt options/train/IMDN.yaml --name IMDN_bs16ps96lr2e-4  --lr 2e-4 --bs 16 --ps 96 --gpu_ids 3 --use_chop
#python train.py --opt options/train/IMDN.yaml --name IMDN_bs16ps96lr2e-4_noskip  --lr 2e-4 --bs 16 --ps 96 --gpu_ids 2 --use_chop
python train.py --opt options/train/CARN.yaml --name CARN_bs16ps64lr1e-4_skip  --lr 1e-4 --bs 16 --ps 64 --gpu_ids 0 --use_chop
