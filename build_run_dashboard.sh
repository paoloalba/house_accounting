
#!/bin/bash

export NOTEBOOK_DIR=./notebooks
export registry=myregistry.io
export versionNumber=beta

export dockerfile_target=dashboard

export dockerfile_src=Dockerfile_dash
export dockerfile_compose_src=docker-compose-dash.yaml
export repositoryName=house_accounting

export PERMANENT_STORAGE=./permanent_storage

docker-compose --project-name ${dockerfile_target} -f ${dockerfile_compose_src} build
docker-compose --project-name ${dockerfile_target} -f ${dockerfile_compose_src} up

# read  -n 1 -p "Input Selection:" mainmenuinput

docker-compose --project-name ${dockerfile_target} -f ${dockerfile_compose_src} down
