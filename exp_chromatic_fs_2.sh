



for config_ in 20 10 5 2 100 
do




python -u train.py --dataset_name blender --root_dir "./datasets/nerf_synthetic/lego" --N_importance 64 --img_wh 400 400 --noise_std 0 --chroma_std "0,0,0" --low_datanum $config_ --num_epochs 16  --batch_size 6144 --optimizer radam --lr 5e-3 --lr_scheduler steplr --decay_step 2 4 8 --num_gpus 1 --decay_gamma 0.5 --exp_name exp_chroma_lego 2>&1 | tee -a chromatic_fs.log

python -u train.py --dataset_name blender --root_dir "./datasets/nerf_synthetic/lego" --N_importance 64 --img_wh 400 400 --noise_std 0 --chroma_std "0.3,0,0" --low_datanum $config_ --num_epochs 16 --batch_size 6144 --optimizer radam --lr 5e-3 --lr_scheduler steplr --decay_step 2 4 8 --num_gpus 1 --decay_gamma 0.5 --exp_name exp_chroma_lego 2>&1 | tee -a chromatic_fs.log

python -u train.py --dataset_name blender --root_dir "./datasets/nerf_synthetic/lego" --N_importance 64 --img_wh 400 400 --noise_std 0 --chroma_std "0.5,0,0" --low_datanum $config_ --num_epochs 16 --batch_size 6144 --optimizer radam --lr 5e-3 --lr_scheduler steplr --decay_step 2 4 8  --num_gpus 1 --decay_gamma 0.5 --exp_name exp_chroma_lego 2>&1 | tee -a chromatic_fs.log

python -u train.py --dataset_name blender --root_dir "./datasets/nerf_synthetic/lego" --N_importance 64 --img_wh 400 400 --noise_std 0 --chroma_std "1.0,0,0" --low_datanum $config_ --num_epochs 16 --batch_size 6144 --optimizer radam --lr 5e-3 --lr_scheduler steplr --decay_step 2 4 8 --num_gpus 1 --decay_gamma 0.5 --exp_name exp_chroma_lego 2>&1 | tee -a chromatic_fs.log



done





