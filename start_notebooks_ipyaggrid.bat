@set chrome_path="C:\Program Files\Google\Chrome\Application\chrome.exe"
@set local_token="plutarco"
@set JUPYTER_CONTAINER_HOST_PORT=8788
@set JUPYTER_CONTAINER_DEBUG_HOST_PORT=8789

start build_run.bat ipyaggrid "exit"

timeout 12

start "" %chrome_path% --new-window http://127.0.0.1:%JUPYTER_CONTAINER_HOST_PORT%/lab?token=%local_token%