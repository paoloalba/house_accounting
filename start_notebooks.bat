@set chrome_path="C:\Program Files\Google\Chrome\Application\chrome.exe"

@set local_token="plutarco"
@set HOST_PORT=8888
@set CONTAINER_PORT=8888
@set DEBUG_HOST_PORT=8889

start build_run.bat prod "exit"

timeout 12

start "" %chrome_path% --new-window http://127.0.0.1:%HOST_PORT%/lab?token=%local_token%