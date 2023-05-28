#!/bin/bash
aggregation_methods=("mean" "mean --oracle_mode" "our_aggr" "fltrust" "our_aggr --injective_florida" "flame" "afa" "geom_median")
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
        for ((k=0; k<2; k++)); do
            tmux split-window -h
            tmux select-layout tiled
            tmux select-pane -t $((j*2+k+1))
            pane_name="${types[$j]} ${attack_methods[$k]}"
            output_dir_name="default_${aggregation_methods[$i]}_${types[$j]}_${attack_methods[$k]}"
            command="python main.py --aggregation_methods=${aggregation_methods[$i]} --type=${types[$j]} --attack_methods=${attack_methods[$k]} --hash=$output_dir_name"
            tmux select-pane -T $pane_name
            tmux send-keys -t $((i+1)).$((j*2+k+1)) "$command" C-m
            tmux select-pane -t $((j*2+k+1))
        done
    done
done

# Attach to the session
tmux attach-session -t mysession
