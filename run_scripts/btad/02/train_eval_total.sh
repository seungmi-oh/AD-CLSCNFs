mp="${mp:-./../models}"
seed_fe="${seed_type:+1708118486}"
seed_nf="${seed_type:+1708220288}"
seed_base="${seed_type:+1707841336}"
echo "Model_root_path "$mp""

if [ -z "$seed_fe" ]
        then python3 main.py --dataset btad -cl 02 -run 6 --is_train yes --model_path $mp -pl 1 2 3 -inp 256 --sub_epochs 5 --meta_epochs 1 --train_type fe_only -bs 16 --skip_connection yes --lr 1e-4 --lr_warm no --lr_warm_epochs 0 --freeze_enc_epochs 7 --repeat_num 3 --loss_type smooth_cls --use_in_domain_data yes 
        else python3 main.py --dataset btad -cl 02 -run 6 --is_train yes --model_path $mp -pl 1 2 3 -inp 256 --sub_epochs 5 --meta_epochs 1 --train_type fe_only -bs 16 --skip_connection yes --lr 1e-4 --lr_warm no --lr_warm_epochs 0 --freeze_enc_epochs 7 --repeat_num 3 --loss_type smooth_cls --use_in_domain_data yes --seed $seed_fe 
fi
python3 main.py --dataset btad -cl 02 -run 6 --is_train no --model_path $mp -pl 1 2 3 -inp 256 --sub_epochs 5 --meta_epochs 1 --infer_type fe_only -bs 32 --skip_connection yes --lr 1e-4 --lr_warm no --lr_warm_epochs 0 --freeze_enc_epochs 7 --repeat_num 3 --loss_type smooth_cls --use_in_domain_data yes --viz yes --pro yes 
if [ -z "$seed_nf" ]
        then python3 main.py --dataset btad -cl 02 -run 6 --is_train yes --model_path $mp -pl 1 2 3 -inp 256 --sub_epochs 6 --meta_epochs 8 --train_type nf_only -bs 32 --skip_connection yes --loss_type smooth_cls --use_in_domain_data yes --lr 2e-5 
        else python3 main.py --dataset btad -cl 02 -run 6 --is_train yes --model_path $mp -pl 1 2 3 -inp 256 --sub_epochs 6 --meta_epochs 8 --train_type nf_only -bs 32 --skip_connection yes --loss_type smooth_cls --use_in_domain_data yes --lr 2e-5 --seed $seed_nf  
fi
python3 main.py --dataset btad -cl 02 -run 6 --is_train no --model_path $mp -pl 1 2 3 -inp 256 --sub_epochs 6 --meta_epochs 8 --infer_type nf_only -bs 32 --skip_connection yes --loss_type smooth_cls --use_in_domain_data yes --viz yes --pro yes 
python3 main.py --dataset btad -cl 02 -run 6 --is_train no --model_path $mp -pl 1 2 3 -inp 256 --sub_epochs 6 --meta_epochs 8 --infer_type joint -bs 32 --skip_connection yes --loss_type smooth_cls --use_in_domain_data yes --viz yes --pro yes --w_fe 0.05 

if [ -z "$seed_base" ]
        then python3 main.py --dataset btad -cl 02 -run 6 --is_train yes --model_path $mp -pl 1 2 3 -inp 256 --sub_epochs 6 --meta_epochs 8 --train_type base-nf_only -bs 32 --skip_connection yes --loss_type smooth_cls --use_in_domain_data yes --lr 2e-5 
        else python3 main.py --dataset btad -cl 02 -run 6 --is_train yes --model_path $mp -pl 1 2 3 -inp 256 --sub_epochs 6 --meta_epochs 8 --train_type base-nf_only -bs 32 --skip_connection yes --loss_type smooth_cls --use_in_domain_data yes --lr 2e-5 --seed $seed_base 
fi
python3 main.py --dataset btad -cl 02 -run 6 --is_train no --model_path $mp -pl 1 2 3 -inp 256 --sub_epochs 6 --meta_epochs 8 --infer_type base-nf_only -bs 32 --skip_connection yes --loss_type smooth_cls --use_in_domain_data yes --viz yes --pro yes  
python3 main.py --dataset btad -cl 02 -run 6 --is_train no --model_path $mp -pl 1 2 3 -inp 256 --sub_epochs 6 --meta_epochs 8 --infer_type joint_not_sharing -bs 32 --skip_connection yes --loss_type smooth_cls --use_in_domain_data yes --viz yes --pro yes --w_fe 0.05

