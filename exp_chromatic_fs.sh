



for config_ in 20 10 5 2 100 
do


times__=$((100/$config_))


echo "config"
echo $config_
echo "times"
echo $times__


python -u train.py --dataset_name blender --root_dir "./datasets/nerf_synthetic/lego" --N_importance 64 --img_wh 400 400 --noise_std 0 --chroma_std "0,0,0" --low_datanum $config_ --num_epochs $((times__*16))  --batch_size 6144 --optimizer radam --lr 5e-3 --lr_scheduler steplr --decay_step $((times__*2)) $((times__*4)) $((times__*8)) --num_gpus 1 --decay_gamma 0.5 --exp_name exp_chroma_lego 2>&1 | tee -a chromatic_fs.log

python -u train.py --dataset_name blender --root_dir "./datasets/nerf_synthetic/lego" --N_importance 64 --img_wh 400 400 --noise_std 0 --chroma_std "0.1,0,0" --low_datanum $config_ --num_epochs $((times__*16)) --batch_size 6144 --optimizer radam --lr 5e-3 --lr_scheduler steplr --decay_step $((times__*2)) $((times__*4)) $((times__*8)) --num_gpus 1 --decay_gamma 0.5 --exp_name exp_chroma_lego 2>&1 | tee -a chromatic_fs.log

python -u train.py --dataset_name blender --root_dir "./datasets/nerf_synthetic/lego" --N_importance 64 --img_wh 400 400 --noise_std 0 --chroma_std "0.1,0.1,0.1" --low_datanum $config_ --num_epochs $((times__*16)) --batch_size 6144 --optimizer radam --lr 5e-3 --lr_scheduler steplr --decay_step $((times__*2)) $((times__*4)) $((times__*8))  --num_gpus 1 --decay_gamma 0.5 --exp_name exp_chroma_lego 2>&1 | tee -a chromatic_fs.log

python -u train.py --dataset_name blender --root_dir "./datasets/nerf_synthetic/lego" --N_importance 64 --img_wh 400 400 --noise_std 0 --chroma_std "0.3,0.1,0.1" --low_datanum $config_ --num_epochs $((times__*16)) --batch_size 6144 --optimizer radam --lr 5e-3 --lr_scheduler steplr --decay_step $((times__*2)) $((times__*4)) $((times__*8))  --num_gpus 1 --decay_gamma 0.5 --exp_name exp_chroma_lego 2>&1 | tee -a chromatic_fs.log



done





