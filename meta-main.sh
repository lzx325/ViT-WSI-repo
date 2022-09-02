#!/usr/bin/env bash
{
yell() { echo "$0: $*" >&2; }
die() {
	yell "$1";
	if ! [ -z ${2+x} ];
	then
		exit  "$2"
	else
		exit 111;
	fi
}
source shell_scripts/setup_environment.sh

operation="$1"
case "$operation" in 
	create_patches)
		source_sh="shell_scripts/create_patches.sh"
		;;
	extract_features)
		source_sh="shell_scripts/extract_features.sh"
		;;
	build_slide_graph)
		source_sh="shell_scripts/build_slide_graph.sh"
		;;
	create_splits)
		source_sh="shell_scripts/create_splits.sh"
		;;
esac

if [ ! -z ${source_sh+x} ]; then
	echo "sourcing configuration shell script: $source_sh"
	source "$source_sh"
fi

case "$operation" in 
	create_patches)
	for patch_size in "${patch_size_list[@]}"; do
		[ -z "$data_files_dir" ] || [ -z "$patch_size" ] && { die "empty options"; }
		echo "==== create_patches ===="
		echo "data_files_dir: $data_files_dir"
		echo "patch_size: $patch_size"
		echo "========================"
		python create_patches_fp.py \
		--source "$data_files_dir/WSI" \
		--save_dir "$data_files_dir/patches/patches--patch_size_$patch_size" \
		--patch_size "$patch_size" --step_size "$patch_size" --preset "$preset" --seg --patch --stitch & 
	done
	;;

	extract_features)

	for patch_size in "${patch_size_list[@]}"; do
		patches_dir="$data_files_dir/patches/patches--patch_size_${patch_size}"
		features_dir="$data_files_dir/features/features--vit_large_patch16_384--patch_size_${patch_size}_384"

		n_jobs=5
		time="12"
		n_cpus="8"
		memory="100"
		name="feature_extraction--$feature_model_type"

		echo "==== extract_features ===="
		echo "data_files_dir: $data_files_dir"
		echo "patches_dir: $patches_dir"
		echo "features_dir: ${features_dir}"
		echo "feature_model_type: $feature_model_type"
		echo "feature_custom_ckpt_fp: $feature_custom_ckpt_fp"
		echo "target_patch_size: $target_patch_size" # TODO: make clear custom_downsample
		echo "n_jobs: $n_jobs"
		echo "=========================="
		for i in $(seq 0 $((n_jobs-1))); do
		mkdir -p "slurm"
		echo "$name"
		if false; then
			sbatch <<- EOF
				#!/bin/bash
				#SBATCH -N 1
				#SBATCH -J ${name}
				#SBATCH --partition=batch
				#SBATCH -o slurm/%J.out
				#SBATCH -e slurm/%J.err
				#SBATCH --time=${time}:00:00
				#SBATCH --mem=${memory}G
				#SBATCH --cpus-per-task=${n_cpus}
				#SBATCH --constraint="[a100]"
				#SBATCH --gres=gpu:1
				#run the application:
				python extract_features_fp.py \
				--data_h5_dir  "$patches_dir" \
				--data_slide_dir  "$data_files_dir/WSI" \
				--csv_path "$patches_dir/process_list_autogen.csv" \
				--feat_dir "$features_dir" \
				--model_type "$feature_model_type" \
				--custom_ckpt_fp "$feature_custom_ckpt_fp" \
				--slide_ext ."$ext" \
				--batch_size 256 \
				--target_patch_size "$target_patch_size" \
				--n_tasks "$n_jobs" \
				--task_id "$i"
			EOF
		else
			python extract_features_fp.py \
					--data_h5_dir  "$patches_dir" \
					--data_slide_dir  "$data_files_dir/WSI" \
					--csv_path "$patches_dir/process_list_autogen.csv" \
					--feat_dir "$features_dir" \
					--model_type "$feature_model_type" \
					--custom_ckpt_fp "$feature_custom_ckpt_fp" \
					--slide_ext ."$ext" \
					--batch_size 256 \
					--target_patch_size "$target_patch_size" \
					--n_tasks "$n_jobs" \
					--task_id "$i"
		fi

		
		done
	done
	;;

	build_slide_graph)

	n_jobs=10
	time="4"
	n_cpus="16"
	memory="100"
	name="build_slide_graph"

	echo "==== build_slide_graph ===="
	echo "n_jobs: $n_jobs"
	echo "data_files_dir: $data_files_dir"
	echo "patches_dir_name: $patches_dir_name"
	echo "features_dir_name: $features_dir_name"
	echo "==========================="

	for((i=0;i<$n_jobs;i++)); do
	mkdir -p "slurm"
	echo "$name"
	if false; then
		sbatch <<- EOF
			#!/bin/bash
			#SBATCH -N 1
			#SBATCH -J ${name}
			#SBATCH --partition=batch
			#SBATCH -o slurm/%J.out
			#SBATCH -e slurm/%J.err
			#SBATCH --time=${time}:00:00
			#SBATCH --mem=${memory}G
			#SBATCH --cpus-per-task=${n_cpus}
			#SBATCH --constraint="[a100]"
			#SBATCH --gres=gpu:1
			#run the application:
			python build_slide_graph.py \
			--n_tasks "$n_jobs" \
			--task_id "$i" \
			--data_files_dir "$data_files_dir" \
			--dataset_name "$dataset_name" \
			--patches_dir_name "$patches_dir_name" \
			--features_dir_name "$features_dir_name" 

		EOF
	else
		python build_slide_graph.py \
			--n_tasks "$n_jobs" \
			--task_id "$i" \
			--data_files_dir "$data_files_dir" \
			--dataset_name "$dataset_name" \
			--patches_dir_name "$patches_dir_name" \
			--features_dir_name "$features_dir_name" 
	fi
	done
	;;

	create_splits)

	echo "==== create_splits_seq ===="
	echo "info_csv_path: $info_csv_path"
	echo "split_dir: $split_dir"
	echo "label_dict_string: $label_dict_string"
	echo "task: $task"
	echo "test_same_as_val: $test_same_as_val"
	echo "==========================="
	python create_splits_seq.py \
		--csv_path "$info_csv_path" \
		--save_dir "$split_dir" \
		--label_dict "$label_dict_string" \
		--task "$task" \
		--seed 1 \
		--label_frac 1 \
		--val_frac 0.2 \
		--test_frac 0.2 \
		--k 10 \
		--test_same_as_val "${test_same_as_val:-false}"
	;;

	

	train_eval)
		exp_code="$2"
		config_fp="config_files/${exp_code}.yaml"
		default_fp="config_files/default.yaml"
		for fold in {0..9}; do
			if false; then
				time="12"
				n_cpus="16"
				memory="100"
				name="$exp_code"
				mkdir -p "slurm"
				sbatch <<- EOF
				#!/bin/bash
				#SBATCH -N 1
				#SBATCH -J ${name}
				#SBATCH --partition=batch
				#SBATCH -o slurm/%J.out
				#SBATCH -e slurm/%J.err
				#SBATCH --time=${time}:00:00
				#SBATCH --mem=${memory}G
				#SBATCH --cpus-per-task=${n_cpus}
				#SBATCH --constraint="[a100]"
				#SBATCH --gres=gpu:1
				set -e
				#run the application:
				python -u train_eval.py train --config "$config_fp" --default "$default_fp" --fold "$fold" 
				python -u train_eval.py evaluate --config "$config_fp" --default "$default_fp" --fold "$fold" 
				EOF
			else
				python -u train_eval.py train --config "$config_fp" --default "$default_fp" --fold "$fold" 
				python -u train_eval.py evaluate --config "$config_fp" --default "$default_fp" --fold "$fold" 
			fi
		done
	;;

	eval_aggr)
		exp_code="$2"
		config_fp="config_files/${exp_code}.yaml"
		default_fp="config_files/default.yaml"
		train_dir="$(cat $config_fp |grep '^train_root_dir'|awk -F'train_root_dir: ' '{print $2}')"
		train_dir="${train_dir#\"}"
		train_dir="${train_dir%\"}"
		echo "using train_dir: $train_dir"
		python eval_aggr.py "$train_dir"/"$exp_code"
	;;

	*)
		echo "unknown option" >&2
		exit 1
	;;
esac

exit 0;
}
