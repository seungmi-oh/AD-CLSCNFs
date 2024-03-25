mp="${mp:-./../best_models}"
echo "Run $0"
echo "Model_root_path "$mp""

python3 main.py --dataset mvtec -cl grid -run 2 --is_train no  -pl 1 2 3  -inp 256 --sub_epoch 2 --meta_epochs 15 --infer_type fe_only  -bs 32 --skip_connection yes  --lr 1e-4  --lr_warm no --lr_warm_epochs 0  --w_fe 1.0 --freeze_enc_epochs 85 --loss_type smooth_cls --model_path $mp --use_in_domain_data yes --viz yes --pro yes 
python3 main.py --dataset mvtec -cl grid -run 2 --is_train no -pl 1 2 3  -inp 256 --sub_epoch 8 --meta_epochs 25 --infer_type nf_only -bs 32 --skip_connection yes --loss_type smooth_cls --model_path $mp  --use_in_domain_data yes --viz yes --pro yes 
python3 main.py --dataset mvtec -cl grid -run 2 --is_train no -pl 1 2 3  -inp 256 --sub_epoch 8 --meta_epochs 25 --infer_type joint -bs 32 --skip_connection yes --loss_type smooth_cls --model_path $mp  --use_in_domain_data yes --viz yes --pro yes --w_fe 1.0  

python3 main.py --dataset mvtec -cl grid -run 2 --is_train no -pl 1 2 3  -inp 256 --sub_epoch 8 --meta_epochs 25 --infer_type base-nf_only -bs 32 --skip_connection yes --loss_type smooth_cls --model_path $mp  --use_in_domain_data yes --viz yes --pro yes 
python3 main.py --dataset mvtec -cl grid -run 2 --is_train no -pl 1 2 3  -inp 256 --sub_epoch 8 --meta_epochs 25 --infer_type joint_not_sharing -bs 32 --skip_connection yes --loss_type smooth_cls --model_path $mp  --use_in_domain_data yes --viz yes --pro yes --w_fe 1.0
