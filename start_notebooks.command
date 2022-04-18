#!/bin/bash
cd "$(dirname "$0")"

export local_token="plutarco"
export JUPYTER_CONTAINER_HOST_PORT=8888

# echo sh build_run.sh prod > aux_cmd.command; chmod +x aux_cmd.command; open aux_cmd.command

sh ./build_run.sh prod &

sleep 12

open -na "Google Chrome" --args --new-window "http://127.0.0.1:${JUPYTER_CONTAINER_HOST_PORT}/lab?token=${local_token}"