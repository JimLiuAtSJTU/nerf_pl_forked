



for config_ in 2 10 5 20 # 5 10 20
do






python -u train.py --dataset_name llff_chromatic --root_dir "./datasets/nerf_llff_data/fern" --N_importance 128  --img_wh 504 378 --chroma_std "0,0,0" --low_datanum $config_   --num_epochs 30 --batch_size 1024 --optimizer adam --lr 5e-4 --lr_scheduler steplr --decay_step 10 20 --num_gpus 2 --decay_gamma 0.5 --exp_name exp_llff 2>&1 | tee -a fern.log

python -u train.py --dataset_name llff_chromatic --root_dir "./datasets/nerf_llff_data/fern" --N_importance 128  --img_wh 504 378 --chroma_std "0.1,0,0" --low_datanum $config_  --num_epochs 30 --batch_size 1024 --optimizer adam --lr 5e-4 --lr_scheduler steplr --decay_step 10 20 --num_gpus 2 --decay_gamma 0.5 --exp_name exp_llff 2>&1 | tee -a fern.log

python -u train.py --dataset_name llff_chromatic --root_dir "./datasets/nerf_llff_data/fern" --N_importance 128  --img_wh 504 378 --chroma_std "0.3,0,0" --low_datanum $config_  --num_epochs 30 --batch_size 1024 --optimizer adam --lr 5e-4 --lr_scheduler steplr --decay_step 10 20 --num_gpus 2 --decay_gamma 0.5 --exp_name exp_llff 2>&1 | tee -a fern.log

python -u train.py --dataset_name llff_chromatic --root_dir "./datasets/nerf_llff_data/fern" --N_importance 128  --img_wh 504 378 --chroma_std "0.5,0,0" --low_datanum $config_  --num_epochs 30 --batch_size 1024 --optimizer adam --lr 5e-4 --lr_scheduler steplr --decay_step 10 20 --num_gpus 2 --decay_gamma 0.5 --exp_name exp_llff 2>&1 | tee -a fern.log




done





