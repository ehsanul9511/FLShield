#!/bin/bash

# Function to generate hash
# Usage: generate_hash "algorithm" "string"
generate_hash() {
    algorithm=$1
    string=$2

    case "$algorithm" in
        "md5")
            hash=$(echo -n "$string" | md5sum | awk '{ print $1 }')
            ;;
        "sha1")
            hash=$(echo -n "$string" | sha1sum | awk '{ print $1 }')
            ;;
        "sha256")
            hash=$(echo -n "$string" | sha256sum | awk '{ print $1 }')
            ;;
        *)
            echo "Unsupported algorithm: $algorithm"
            return 1
            ;;
    esac

    echo "$hash"
}

# Example usage
aggregation_methods=("mean" "our_aggr")

# alias sha256sum="shasum -a 256"
export PATH="/scratch/ejk5818/miniconda3/bin/python:$PATH"
source /scratch/ejk5818/miniconda3/bin/activate fl

for str in "${aggregation_methods[@]}"; do
    command="python main.py --aggregation_method=\"$str\" --type=cifar"
    echo "$command"
    sha256_hash=$(echo -n "$command" | sha256sum | awk '{ print $1 }')
    echo "SHA-256 hash for \"$str\": $sha256_hash"

    $command &

    echo
done

echo "Waiting"

wait

echo "Done"
