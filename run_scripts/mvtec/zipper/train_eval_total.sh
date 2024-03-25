mp="${mp:-./../models}"
seed_fe="${seed_type:+1690590864}"
seed_nf="${seed_type:+1690595769}"
seed_base="${seed_type:+1690274817}"
echo "Model_root_path "$mp""

if [ -z "$seed_fe" ]
        then python3 main.py --dataset mvtec -cl zipper -run 1 --is_train yes -pl 1 2 3 -inp 256 --sub_epochs 2 --meta_epochs 20 --train_type fe_only -bs 16 --skip_connection yes --lr 1e-4 --lr_warm no --lr_warm_epochs 0 --freeze_enc_epochs 80 --repeat_num 5 --loss_type smooth_cls --model_path $mp --use_in_domain_data yes  
        else python3 main.py --dataset mvtec -cl zipper -run 1 --is_train yes -pl 1 2 3 -inp 256 --sub_epochs 2 --meta_epochs 20 --train_type fe_only -bs 16 --skip_connection yes --lr 1e-4 --lr_warm no --lr_warm_epochs 0 --freeze_enc_epochs 80 --repeat_num 5 --loss_type smooth_cls --model_path $mp --use_in_domain_data yes --seed $seed_fe 
fi
python3 main.py --dataset mvtec -cl zipper -run 1 --is_train no -pl 1 2 3 -inp 256 --sub_epochs 2 --meta_epochs 20 --infer_type fe_only -bs 32 --skip_connection yes --lr 1e-4 --lr_warm no --lr_warm_epochs 0 --freeze_enc_epochs 80 --repeat_num 5 --loss_type smooth_cls --model_path $mp --use_in_domain_data yes --viz yes --pro yes 
if [ -z "$seed_nf" ]
        then python3 main.py --dataset mvtec -cl zipper -run 1 --is_train yes -pl 1 2 3 -inp 256 --sub_epochs 8 --meta_epochs 25 --train_type nf_only -bs 32 --skip_connection yes --loss_type smooth_cls --model_path $mp --use_in_domain_data yes 
        else python3 main.py --dataset mvtec -cl zipper -run 1 --is_train yes -pl 1 2 3 -inp 256 --sub_epochs 8 --meta_epochs 25 --train_type nf_only -bs 32 --skip_connection yes --loss_type smooth_cls --model_path $mp --use_in_domain_data yes --seed $seed_nf 
fi
python3 main.py --dataset mvtec -cl zipper -run 1 --is_train no -pl 1 2 3 -inp 256 --sub_epochs 8 --meta_epochs 25 --infer_type nf_only -bs 32 --skip_connection yes --loss_type smooth_cls --model_path $mp --use_in_domain_data yes --viz yes --pro yes 
python3 main.py --dataset mvtec -cl zipper -run 1 --is_train no -pl 1 2 3 -inp 256 --sub_epochs 8 --meta_epochs 25 --infer_type joint -bs 32 --skip_connection yes --loss_type smooth_cls --model_path $mp --use_in_domain_data yes --viz yes --pro yes --w_fe 0.15 

if [ -z "$seed_base" ]
        then python3 main.py --dataset mvtec -cl zipper -run 1 --is_train yes -pl 1 2 3 -inp 256 --sub_epochs 8 --meta_epochs 25 --train_type base-nf_only -bs 32 --skip_connection yes --loss_type smooth_cls --model_path $mp --use_in_domain_data yes 
        else python3 main.py --dataset mvtec -cl zipper -run 1 --is_train yes -pl 1 2 3 -inp 256 --sub_epochs 8 --meta_epochs 25 --train_type base-nf_only -bs 32 --skip_connection yes --loss_type smooth_cls --model_path $mp --use_in_domain_data yes --seed $seed_base 
fi
python3 main.py --dataset mvtec -cl zipper -run 1 --is_train no -pl 1 2 3 -inp 256 --sub_epochs 8 --meta_epochs 25 --infer_type base-nf_only -bs 32 --skip_connection yes --loss_type smooth_cls --model_path $mp --use_in_domain_data yes --viz yes --pro yes 
python3 main.py --dataset mvtec -cl zipper -run 1 --is_train no -pl 1 2 3 -inp 256 --sub_epochs 8 --meta_epochs 25 --infer_type joint_not_sharing -bs 32 --skip_connection yes --loss_type smooth_cls --model_path $mp --use_in_domain_data yes --viz yes --pro yes --w_fe 0.15
