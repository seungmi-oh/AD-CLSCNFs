btad_classes="01 02 03"
seed_type=""

mp="./../models"
for cl in $btad_classes;
do
	echo "run_scripts/btad/$cl/train_eval_total.sh"
	source run_scripts/btad/$cl/train_eval_total.sh 
done

infer_types="fe_only nf_only joint base-nf_only joint_not_sharing"

for inf_type in $infer_types;
do
	python3 parse_results.py --dataset btad -pl 1 2 3  -inp 256 --infer_type $inf_type --loss_type smooth_cls --model_path $mp --use_in_domain_data yes 
done

