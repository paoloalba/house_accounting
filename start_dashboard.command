#!/bin/bash
export cur_dir_name="$(dirname "$0")"
cd "${cur_dir_name}"

export HOST_PORT=8050
export CONTAINER_PORT=8050
export DEBUG_HOST_PORT=8889

export aux_script_name="aux_cmd.command"

echo -e "export CONTAINER_PORT=${CONTAINER_PORT}\nexport HOST_PORT=${HOST_PORT}\nexport DEBUG_HOST_PORT=${DEBUG_HOST_PORT}\ncd ${cur_dir_name}\nsh build_run.sh Dockerfile_dash dashboard" > ${aux_script_name}; chmod +x ${aux_script_name}; open ${aux_script_name}

sleep 4

open -na "Google Chrome" --args --new-window "http://127.0.0.1:${HOST_PORT}"

rm -f ${aux_script_name}
