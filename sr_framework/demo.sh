#python train.py --opt options/train/CARN.yaml --name CARN_bs16ps64lr1e-4_x2  --scale 2 --lr 1e-4 --bs 16 --ps 64 --gpu_ids 0 --use_chop
#python train.py --opt options/train/RCAN.yaml --name RCAN_bs16ps48lr1e-4_x2 --scale 2 --lr 1e-4 --bs 16 --ps 48 --gpu_ids 2 --use_chop
python train.py --opt options/train/MemNet.yaml --name MemNet_bs16ps48lr1e-4_x2 --scale 2 --lr 1e-4 --bs 16 --ps 48 --gpu_ids 3 --use_chop
