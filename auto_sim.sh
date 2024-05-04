#!/bin/bash

# To Execute:
# ./auto_sim.sh

start_display=1000
end_display=1005
simnum=0

for ((display=start_display; display<=end_display; display++)); do
    # Set the DISPLAY environment variable
    export DISPLAY=:$display

    # Start the first command in the background
    wine pika.exe &
    # Start the second command in the background
    x11vnc -display :$display &

    # Run the third command in a subshell to keep it independent
    (
        # Function to start and monitor the third command
        function keep_running {
            while true; do
                python actor.py --simnum $simnum
                # Check exit status
                if [ $? -ne 0 ]; then
                    echo "actor.py crashed with exit code $?. Restarting..."
                    sleep 1  # pause before restarting to prevent spamming
                else
                    break  # exit loop if script ends successfully
                fi
            done
        }
        # Call the function in background to allow next loop iteration to start
        keep_running &
    ) &

    # Increment simnum for next iteration
    ((simnum++))
done

# Wait for all background processes to finish
wait
