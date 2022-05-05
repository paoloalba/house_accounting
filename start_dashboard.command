#!/bin/bash
export cur_dir_name="$(dirname "$0")"
cd "${cur_dir_name}"

export aux_script_name="aux_cmd.command"

echo -e "cd ${cur_dir_name}\nsh build_run_dashboard.sh" > ${aux_script_name}; chmod +x ${aux_script_name}; open ${aux_script_name}

sleep 4

open -na "Google Chrome" --args --new-window "http://127.0.0.1:8050"

rm -f ${aux_script_name}
