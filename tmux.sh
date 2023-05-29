#!/bin/bash
aggregation_methods=("mean" "mean --oracle_mode" "our_aggr" "our_aggr --injective_florida" "fltrust" "flame" "afa" "geom_median")
types=("cifar" "emnist" "fmnist")
attack_methods=("targeted_label_flip" "dba")

# Create 8 tmux windows
tmux new-session -d -s mysession

for ((i=0; i<8; i++)); do
    # Create a new window
    tmux new-window -t mysession:$(($i+1))
    tmux rename-window -t mysession:$(($i+1)) "echo ${aggregation_methods[$i]}"

    # Split the window into 4x4 panes
    for ((j=0; j<3; j++)); do
        for ((k=0; k<1; k++)); do
            tmux split-window -h
            tmux select-layout tiled
            tmux select-pane -t $((j*1+k))
            pane_name="${types[$j]} ${attack_methods[$k]}"
            output_dir_name="default_${aggregation_methods[$i]}_${types[$j]}_${attack_methods[$k]}"
            output_dir_name=$(echo "$output_dir_name" | sed 's/ //g')
            command="CUDA_VISIBLE_DEVICES=$i python main.py --aggregation_methods=${aggregation_methods[$i]} --type=${types[$j]} --attack_methods=${attack_methods[$k]} --hash=$output_dir_name"
            # tmux select-pane -T $pane_name
            tmux send-keys -t $((i+1)).$((j*1+k)) "$command" C-m
            tmux select-pane -t $((j*1+k))
        done
    done
done

# Attach to the session
tmux attach-session -t mysession
