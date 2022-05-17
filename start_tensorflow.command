#!/bin/bash
export cur_dir_name="$(dirname "$0")"
cd "${cur_dir_name}"

export local_token="plutarco"
export HOST_PORT=8891
export CONTAINER_PORT=8888
export DEBUG_HOST_PORT=8890

export aux_script_name="aux_cmd.command"

echo -e "export CONTAINER_PORT=${CONTAINER_PORT}\nexport HOST_PORT=${HOST_PORT}\nexport DEBUG_HOST_PORT=${DEBUG_HOST_PORT}\ncd ${cur_dir_name}\nsh build_run.sh Dockerfile_tensorflow tf" > ${aux_script_name}; chmod +x ${aux_script_name}; open ${aux_script_name}

sleep 4

open -na "Google Chrome" --args --new-window "http://127.0.0.1:${HOST_PORT}/lab?token=${local_token}"

rm -f ${aux_script_name}
