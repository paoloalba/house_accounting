
@set NOTEBOOK_DIR=./notebooks
@set registry=myregistry.io
@set versionNumber=beta

@REM @set dockerfile_target=debug
@REM @set dockerfile_target=prod
@set dockerfile_target=%1

@set dockerfile_src=Dockerfile
@set repositoryName=house_accounting

@set PERMANENT_STORAGE=.\permanent_storage

call docker-compose --project-name %dockerfile_target% build
call docker-compose --project-name %dockerfile_target% up
call docker-compose --project-name %dockerfile_target% down

@if "%2"==""exit"" exit
