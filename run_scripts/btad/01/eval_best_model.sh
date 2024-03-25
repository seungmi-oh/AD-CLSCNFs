mp="${mp:-./../best_models}"
echo "Run $0"
echo "Model_root_path "$mp""

# fe only
python3 main.py --dataset btad -cl 01 -run 3 --is_train no --model_path $mp -pl 1 2 3 -inp 256 --sub_epochs 1 --meta_epochs 10 --infer_type fe_only -bs 32 --skip_connection yes --lr 5e-4 --lr_warm no --lr_warm_epochs 0 --freeze_enc_epochs 70 --repeat_num 2 --loss_type smooth_cls --use_in_domain_data yes --viz yes --pro yes 

# nf only
python3 main.py --dataset btad -cl 01 -run 3 --is_train no --model_path $mp -pl 1 2 3 -inp 256 --sub_epochs 6 --meta_epochs 8 --infer_type nf_only -bs 32 --skip_connection yes --loss_type smooth_cls --use_in_domain_data yes --viz yes --pro yes 

# joint
python3 main.py --dataset btad -cl 01 -run 3 --is_train no --model_path $mp -pl 1 2 3 -inp 256 --sub_epochs 6 --meta_epochs 8 --infer_type joint -bs 32 --skip_connection yes --loss_type smooth_cls --use_in_domain_data yes --viz yes --pro yes --w_fe 0.1

# base nf only
python3 main.py --dataset btad -cl 01 -run 3 --is_train no --model_path $mp -pl 1 2 3 -inp 256 --sub_epochs 6 --meta_epochs 8 --infer_type base-nf_only -bs 32 --skip_connection yes --loss_type smooth_cls --use_in_domain_data yes --viz yes --pro yes  

# joint not sharing feature extractors
python3 main.py --dataset btad -cl 01 -run 3 --is_train no --model_path $mp -pl 1 2 3 -inp 256 --sub_epochs 6 --meta_epochs 8 --infer_type joint_not_sharing -bs 32 --skip_connection yes --loss_type smooth_cls --use_in_domain_data yes --viz yes --pro yes --w_fe 0.1
