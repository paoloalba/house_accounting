#!/bin/bash
export cur_dir_name="$(dirname "$0")"
cd "${cur_dir_name}"

export local_token="plutarco"
export JUPYTER_CONTAINER_HOST_PORT=8888
export aux_script_name="aux_cmd.command"

echo -e "export JUPYTER_CONTAINER_HOST_PORT=${JUPYTER_CONTAINER_HOST_PORT}\ncd ${cur_dir_name}\nsh build_run.sh prod" > ${aux_script_name}; chmod +x ${aux_script_name}; open ${aux_script_name}

sleep 4

open -na "Google Chrome" --args --new-window "http://127.0.0.1:${JUPYTER_CONTAINER_HOST_PORT}/lab?token=${local_token}"

rm -f ${aux_script_name}
