param([switch]$Baseline, [switch]$Trained)
Set-Location -LiteralPath $PSScriptRoot
$simulationPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $simulationPython)) {
    throw 'Install Python and dependencies using the README instructions first.'
}
if ($Trained -and -not $Baseline -and (Test-Path 'models/four-dof-v1/tongue_a_ppo.zip') -and (Test-Path 'models/four-dof-v1/tongue_b_ppo.zip')) {
    & $simulationPython -m scripts.demo --trained
} else {
    & $simulationPython -m scripts.demo
}
