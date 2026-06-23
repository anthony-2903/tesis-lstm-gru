$ErrorActionPreference = "Stop"

$base = "http://127.0.0.1:8000"
$paths = @(
  "/api/health",
  "/api/dashboard",
  "/api/analysis",
  "/api/comparison",
  "/api/history",
  "/api/xai",
  "/api/ai-analysis?type=general"
)

foreach ($path in $paths) {
  $url = "$base$path"
  $response = Invoke-WebRequest -Uri $url -UseBasicParsing
  Write-Host "$($response.StatusCode) $url"
}
