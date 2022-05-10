#!/bin/bash

export NOTEBOOK_DIR=./notebooks
export registry=myregistry.io
export versionNumber=beta

export dockerfile_src=$1
export dockerfile_target=$2

export repositoryName=house_accounting

export PERMANENT_STORAGE=./permanent_storage

docker-compose --project-name ${dockerfile_target} build
docker-compose --project-name ${dockerfile_target} up

# read  -n 1 -p "Input Selection:" mainmenuinput

docker-compose --project-name ${dockerfile_target} down
