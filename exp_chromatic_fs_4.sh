



for config_ in 2 10 # 5 10 20
do




python -u train.py --dataset_name blender --root_dir "./datasets/nerf_synthetic/lego" --N_importance 64 --img_wh 400 400 --noise_std 0 --chroma_std "0,0,0" --low_datanum $config_ --num_epochs 16  --batch_size 2048 --optimizer adam --lr 5e-4 --lr_scheduler steplr --decay_step 2 4 8 --num_gpus 2 --decay_gamma 0.5 --exp_name exp_chroma_lego 2>&1 | tee -a chromatic_fs4.log

#python -u train.py --dataset_name blender --root_dir "./datasets/nerf_synthetic/lego" --N_importance 64 --img_wh 200 200 --noise_std 0 --chroma_std "1.0,0,0" --low_datanum $config_ --num_epochs 16 --batch_size 2048 --optimizer adam --lr 5e-4 --lr_scheduler steplr --decay_step 2 4 8 --num_gpus 2 --decay_gamma 0.5 --exp_name exp_chroma_lego 2>&1 | tee -a chromatic_fs4.log

python -u train.py --dataset_name blender --root_dir "./datasets/nerf_synthetic/lego" --N_importance 64 --img_wh 400 400 --noise_std 0 --chroma_std "0.3,0,0" --low_datanum $config_ --num_epochs 16 --batch_size 2048 --optimizer adam --lr 5e-4 --lr_scheduler steplr --decay_step 2 4 8 --num_gpus 2 --decay_gamma 0.5 --exp_name exp_chroma_lego 2>&1 | tee -a chromatic_fs4.log

#python -u train.py --dataset_name blender --root_dir "./datasets/nerf_synthetic/lego" --N_importance 64 --img_wh 200 200 --noise_std 0 --chroma_std "0.5,0,0" --low_datanum $config_ --num_epochs 16 --batch_size 2048 --optimizer adam --lr 5e-4 --lr_scheduler steplr --decay_step 2 4 8  --num_gpus 2 --decay_gamma 0.5 --exp_name exp_chroma_lego 2>&1 | tee -a chromatic_fs4.log




done





