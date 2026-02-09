#!/bin/bash

# Function to check if a screen session exists and either execute in the existing session or create a new one
start_or_attach_screen_session() {
    local session_name=$1
    local command=$2

    if screen -list | grep -q "\.${session_name}"; then
        echo "Screen session ${session_name} already exists, executing command..."
        screen -S ${session_name} -X stuff "${command}$(printf \\r)"
    else
        echo "Screen session ${session_name} does not exist, creating and starting..."
        screen -dmS ${session_name} bash -c "${command}; exec bash"
    fi
}

start_or_attach_screen_session "session1" "python feature/sampled_attention_mean_revised.py --sample_size=1 --device=cuda:0 --sample_mode=diversified"
start_or_attach_screen_session "session2" "python feature/sampled_attention_mean_revised.py --sample_size=2 --device=cuda:1 --sample_mode=diversified"
start_or_attach_screen_session "session3" "python feature/sampled_attention_mean_revised.py --sample_size=5 --device=cuda:2 --sample_mode=diversified"
start_or_attach_screen_session "session4" "python feature/sampled_attention_mean_revised.py --sample_size=10 --device=cuda:3 --sample_mode=diversified"

echo "All screen sessions processed."
