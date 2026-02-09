
@echo off
timeout /t 2 > nul
start "" http://127.0.0.1:5000
mlflow ui
