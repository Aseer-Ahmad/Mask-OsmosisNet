#!/bin/bash

# Destination directory for synchronized files
DEST_DIR="ssh_output/"
RSYNC_OPTS="-progress -pthrvz"

# Function to sync job outputs
sync_job_outputs() {
  local job_id=$1
  echo "Syncing outputs for Job ID: $job_id"
  rsync $RSYNC_OPTS -e "condor_ssh_to_job" "${job_id}:outputs/*" "$DEST_DIR"
}

# Function to get running job IDs
get_running_jobs() {
  condor_q | awk 'NR > 2 && $7 == "1" { print $10 }'
}

# Main loop to continually sync outputs
while true; do
  echo "Checking for running jobs..."
  running_jobs=$(get_running_jobs)

  if [[ -z $running_jobs ]]; then
    echo "No running jobs found. Killing this job now"
    exit 0
  else
    for job_id in $running_jobs; do
      sync_job_outputs "$job_id"
    done
  fi

  # Wait for 10 minutes before next sync
  sleep 600
done
