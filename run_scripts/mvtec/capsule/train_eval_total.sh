mp="${mp:-./../models}"
seed_fe="${seed_type:+1690608780}"
seed_nf="${seed_type:+1690612390}"
seed_base="${seed_type:+1690179525}"
echo "Model_root_path "$mp""

if [ -z "$seed_fe" ]
        then python3 main.py --dataset mvtec -cl capsule -run 7 --is_train yes -pl 1 2 3 -inp 256 --sub_epochs 2 --meta_epochs 15 --train_type fe_only -bs 32 --skip_connection yes --lr 5e-4 --lr_warm no --lr_warm_epochs 0 --freeze_enc_epochs 85 --loss_type smooth_cls --repeat_num 2 --model_path $mp --use_in_domain_data yes 
        else python3 main.py --dataset mvtec -cl capsule -run 7 --is_train yes -pl 1 2 3 -inp 256 --sub_epochs 2 --meta_epochs 15 --train_type fe_only -bs 32 --skip_connection yes --lr 5e-4 --lr_warm no --lr_warm_epochs 0 --freeze_enc_epochs 85 --loss_type smooth_cls --repeat_num 2 --model_path $mp --use_in_domain_data yes --seed $seed_fe 
fi
python3 main.py --dataset mvtec -cl capsule -run 7 --is_train no -pl 1 2 3 -inp 256 --sub_epochs 2 --meta_epochs 15 --infer_type fe_only -bs 32 --skip_connection yes --lr 5e-4 --lr_warm no --lr_warm_epochs 0  --freeze_enc_epochs 85 --loss_type smooth_cls --repeat_num 2 --model_path $mp --use_in_domain_data yes --viz yes --pro yes 
if [ -z "$seed_nf" ]
        then python3 main.py --dataset mvtec -cl capsule -run 7 --is_train yes -pl 1 2 3 -inp 256 --sub_epochs 8 --meta_epochs 25 --train_type nf_only -bs 32 --skip_connection yes --loss_type smooth_cls --model_path $mp --use_in_domain_data yes 
        else python3 main.py --dataset mvtec -cl capsule -run 7 --is_train yes -pl 1 2 3 -inp 256 --sub_epochs 8 --meta_epochs 25 --train_type nf_only -bs 32 --skip_connection yes --loss_type smooth_cls --model_path $mp --use_in_domain_data yes --seed $seed_nf 
fi
python3 main.py --dataset mvtec -cl capsule -run 7 --is_train no -pl 1 2 3 -inp 256 --sub_epochs 8 --meta_epochs 25 --infer_type nf_only -bs 32 --skip_connection yes --loss_type smooth_cls --model_path $mp --use_in_domain_data yes --viz yes --pro yes 
python3 main.py --dataset mvtec -cl capsule -run 7 --is_train no -pl 1 2 3 -inp 256 --sub_epochs 8 --meta_epochs 25 --infer_type joint -bs 32 --skip_connection yes --loss_type smooth_cls --model_path $mp --use_in_domain_data yes --viz yes --pro yes --w_fe 0.3  

if [ -z "$seed_base" ]
        then python3 main.py --dataset mvtec -cl capsule -run 7 --is_train yes -pl 1 2 3 -inp 256 --sub_epochs 8 --meta_epochs 25 --train_type base-nf_only -bs 32 --skip_connection yes --loss_type smooth_cls --model_path $mp --use_in_domain_data yes 
        else python3 main.py --dataset mvtec -cl capsule -run 7 --is_train yes -pl 1 2 3 -inp 256 --sub_epochs 8 --meta_epochs 25 --train_type base-nf_only -bs 32 --skip_connection yes --loss_type smooth_cls --model_path $mp --use_in_domain_data yes --seed $seed_base 
fi
python3 main.py --dataset mvtec -cl capsule -run 7 --is_train no -pl 1 2 3 -inp 256 --sub_epochs 8 --meta_epochs 25 --infer_type base-nf_only -bs 32 --skip_connection yes --loss_type smooth_cls --model_path $mp --use_in_domain_data yes --viz yes --pro yes 
python3 main.py --dataset mvtec -cl capsule -run 7 --is_train no -pl 1 2 3 -inp 256 --sub_epochs 8 --meta_epochs 25 --infer_type joint_not_sharing -bs 32 --skip_connection yes --loss_type smooth_cls --model_path $mp --use_in_domain_data yes --viz yes --pro yes --w_fe 0.3  
