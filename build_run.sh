
#!/bin/bash

export NOTEBOOK_DIR=./notebooks
export registry=myregistry.io
export versionNumber=beta

export dockerfile_target=$1

export dockerfile_src=Dockerfile
export repositoryName=house_accounting

export PERMANENT_STORAGE=./permanent_storage

docker-compose build
docker-compose up
docker-compose down
