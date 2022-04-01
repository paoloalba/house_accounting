
@set NOTEBOOK_DIR=./notebooks
@set registry=myregistry.io
@set versionNumber=beta

@REM @set dockerfile_target=debug
@REM @set dockerfile_target=prod
@set dockerfile_target=%1

@set dockerfile_src=Dockerfile
@set repositoryName=house_accounting

@set PERMANENT_STORAGE=.\permanent_storage

call docker-compose build
call docker-compose up
call docker-compose down

@if "%2"==""exit"" exit
