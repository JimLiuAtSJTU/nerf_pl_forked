

python -u train.py --dataset_name blender --root_dir "./datasets/nerf_synthetic/lego" --N_importance 64 --img_wh 800 800 --noise_std 0 --chroma_std "0,0,0" --num_epochs 80 --batch_size 2048 --optimizer radam --lr 5e-3 --lr_scheduler steplr --decay_step 2 4 8 --num_gpus 2 --decay_gamma 0.5 --exp_name exp_chroma_lego 2>&1 | tee -a chromatic.log

python -u train.py --dataset_name blender --root_dir "./datasets/nerf_synthetic/lego" --N_importance 64 --img_wh 800 800 --noise_std 0 --chroma_std "0.1,0,0" --num_epochs 80 --batch_size 2048 --optimizer radam --lr 5e-3 --lr_scheduler steplr --decay_step 2 4 8 --num_gpus 2 --decay_gamma 0.5 --exp_name exp_chroma_lego 2>&1 | tee -a chromatic.log

python -u train.py --dataset_name blender --root_dir "./datasets/nerf_synthetic/lego" --N_importance 64 --img_wh 800 800 --noise_std 0 --chroma_std "0.1,0.1,0.1" --num_epochs 80 --batch_size 2048 --optimizer radam --lr 5e-3 --lr_scheduler steplr --decay_step 2 4 8 --num_gpus 2 --decay_gamma 0.5 --exp_name exp_chroma_lego 2>&1 | tee -a chromatic.log

python -u train.py --dataset_name blender --root_dir "./datasets/nerf_synthetic/lego" --N_importance 64 --img_wh 800 800 --noise_std 0 --chroma_std "0.3,0.1,0.1" --num_epochs 80 --batch_size 2048 --optimizer radam --lr 5e-3 --lr_scheduler steplr --decay_step 2 4 8 --num_gpus 2 --decay_gamma 0.5 --exp_name exp_chroma_lego 2>&1 | tee -a chromatic.log


