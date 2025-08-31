#!/bin/bash

# Exit if no arguments provided
if [ "$#" -eq 0 ]; then
    echo "Error: split_method argument is required (e.g. split_method=fold_random_5)"
    exit 1
fi

# Initialize variable
split_method=""

for arg in "$@"; do
    case $arg in
        split_method=*)
            split_method="${arg#*=}"
            shift
            ;;
        *)
            echo "Unknown argument: $arg"
            exit 1
            ;;
    esac
done


# Check if split_method was set
if [ -z "$split_method" ]; then
    echo "Error: split_method not provided (use split_method=your_value)"
    exit 1
fi

# Define allowed split methods
allowed_split_methods=("fold_rand_multiples" "fold_random_5" "fold_modulo_5" "fold_contiguous_5")

# Check if split_method is valid
is_valid=false
for method in "${allowed_split_methods[@]}"; do
    if [ "$split_method" = "$method" ]; then
        is_valid=true
        break
    fi
done

if [ "$is_valid" = false ]; then
    echo "Error: Invalid split_method '$split_method'"
    echo "Allowed values are: ${allowed_split_methods[*]}"
    exit 1
fi

for llm in prosst esm1v; do
    # Set max index based on split_method
    if [ "$split_method" = "fold_rand_multiples" ]; then
        max_idx=68
    else   # "fold_random_5", "fold_modulo_5", "fold_contiguous_5"
        max_idx=216
    fi
    for ((i=0; i<=max_idx; i++)); do
        echo -e "\n\nRunning DMS_idx=$i with llm=$llm and split_method=$split_method\n-----"
        python pgym_cv_benchmark.py split_method=$split_method DMS_idx=$i llm=$llm
        find ./model_saves/ -type f -name '*.pt' -delete  # Delete ProSST pt model checkpoints
    done
done
