param(
  [string]$Mode = "sample"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv")) {
  $python = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
  if (-not (Test-Path $python)) {
    $python = "python"
  }
  & $python -m venv .venv
}

.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m app.pipeline --mode $Mode
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
