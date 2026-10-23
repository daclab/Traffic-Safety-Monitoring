# Run from Windows PowerShell. Copies explicit files to explicit remote paths.
[CmdletBinding()]
param(
    [string]$Serial = '',
    [string]$BoardRoot = '/home/arduino/unoq-yolo11n-ladder',
    [string]$Coco300Source = '',
    [switch]$SkipInstall,
    [switch]$ValidateOnly
)
$ErrorActionPreference = 'Stop'
$Repo = $PSScriptRoot
if ($BoardRoot -notmatch '^/[A-Za-z0-9_./-]+$' -or $BoardRoot -match '(^|/)\.\.(/|$)') {
    throw 'BoardRoot must be an absolute Linux path without spaces or parent traversal.'
}
if (-not $Coco300Source) { $Coco300Source = Join-Path $Repo 'data\coco300' }
$Coco300Source = (Resolve-Path -LiteralPath $Coco300Source).Path
$Expected = Get-Content -Raw -LiteralPath (Join-Path $Repo 'data\coco300\manifest.json') | ConvertFrom-Json
$Actual = Get-Content -Raw -LiteralPath (Join-Path $Coco300Source 'manifest.json') | ConvertFrom-Json
if ($Actual.images.Count -ne 300 -or $Expected.images.Count -ne 300) { throw 'Expected 300 manifest entries.' }
for ($i=0; $i -lt 300; $i++) {
    foreach ($field in @('id','file_name','width','height')) {
        if ($Actual.images[$i].$field -ne $Expected.images[$i].$field) {
            throw "Dataset manifest differs from the fixed COCO-300 protocol at entry $i."
        }
    }
    $name = $Expected.images[$i].file_name
    if ([IO.Path]::GetFileName($name) -ne $name -or $name -notmatch '^\d{12}\.jpg$') { throw "Invalid image name: $name" }
    $image = Join-Path $Coco300Source "images\$name"
    if (-not (Test-Path -LiteralPath $image -PathType Leaf) -or (Get-Item -LiteralPath $image).Length -eq 0) {
        throw "Missing COCO image: $image. Prepare data first or pass -Coco300Source pointing to an existing complete subset."
    }
}
$Models = @('yolo11n_silu_640_float32.tflite','yolo11n_silu_320_float32.tflite','yolo11n_silu_320_w8a8_int8io.tflite')
foreach ($name in $Models) {
    $path = Join-Path $Repo "models\$name"
    if (-not (Test-Path -LiteralPath $path -PathType Leaf) -or (Get-Item -LiteralPath $path).Length -eq 0) {
        throw "Missing model: $path"
    }
}
# Validate runtime payload locally before contacting a board.
$App = Join-Path $Repo 'arduino_app\traffic_safety_monitoring_applab'
foreach ($name in @($Models[0],$Models[2])) {
    $bundle = Join-Path $App "models\$name"
    if (-not (Test-Path -LiteralPath $bundle) -or
        (Get-FileHash -LiteralPath $bundle).Hash -ne (Get-FileHash -LiteralPath (Join-Path $Repo "models\$name")).Hash) {
        throw "App model missing or different from ladder artifact: $bundle"
    }
}
Write-Host 'Local checks passed: three ladder models, two app models, and all 300 expected JPEGs.'
if ($ValidateOnly) { return }
if (-not (Get-Command adb -ErrorAction SilentlyContinue)) { throw 'Install Android platform tools and reopen PowerShell so adb is on PATH.' }
$listing = & adb devices
if ($LASTEXITCODE -ne 0) { throw 'adb devices failed.' }
$connected = @($listing | ForEach-Object { if ($_ -match '^(\S+)\s+device\s*$') { $Matches[1] } })
if (-not $Serial) {
    if ($connected.Count -ne 1) { throw "Specify -Serial from adb devices. Ready devices: $($connected -join ', ')" }
    $Serial = $connected[0]
}
if ($Serial -notin $connected) { throw "Device $Serial is not ready in adb devices." }
function Invoke-Adb {
    & adb -s $Serial @args
    if ($LASTEXITCODE -ne 0) { throw "ADB command failed: $args" }
}
Write-Host "Target: $Serial -> $BoardRoot"
Write-Host 'Stop running ladder/app processes before deployment. Existing results and app calibration are preserved.'

$Payload = @()
foreach ($dir in @('ladder','arduino_app\traffic_safety_monitoring_applab')) {
    foreach ($file in Get-ChildItem -LiteralPath (Join-Path $Repo $dir) -Recurse -File) {
        $relative = $file.FullName.Substring($Repo.Length + 1).Replace('\','/')
        if ($relative -match '/(__pycache__|data)/' -or $file.Extension -eq '.pyc') { continue }
        $Payload += [pscustomobject]@{Local=$file.FullName; Remote="$BoardRoot/$relative"}
    }
}
foreach ($name in $Models) {
    $Payload += [pscustomobject]@{Local=(Join-Path $Repo "models\$name"); Remote="$BoardRoot/models/$name"}
}
$Payload += [pscustomobject]@{Local=(Join-Path $Coco300Source 'manifest.json'); Remote="$BoardRoot/data/coco300/manifest.json"}
foreach ($item in $Expected.images) {
    $Payload += [pscustomobject]@{Local=(Join-Path $Coco300Source "images\$($item.file_name)"); Remote="$BoardRoot/data/coco300/images/$($item.file_name)"}
}
$directories = @($Payload | ForEach-Object { $_.Remote.Substring(0,$_.Remote.LastIndexOf('/')) } | Sort-Object -Unique)
foreach ($dir in $directories) { Invoke-Adb shell "mkdir -p $dir" | Out-Null }
Invoke-Adb shell "mkdir -p $BoardRoot/arduino_app/traffic_safety_monitoring_applab/data" | Out-Null
$checksums = @()
$count = 0
foreach ($entry in $Payload) {
    Invoke-Adb push $entry.Local $entry.Remote | Out-Null
    $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $entry.Local).Hash.ToLowerInvariant()
    $relative = $entry.Remote.Substring($BoardRoot.Length + 1)
    $checksums += "$hash  $relative"
    $count++
    if ($count % 50 -eq 0) { Write-Host "Copied $count / $($Payload.Count) files" }
}
$tempChecksum = Join-Path ([IO.Path]::GetTempPath()) ([IO.Path]::GetRandomFileName())
try {
    [IO.File]::WriteAllText($tempChecksum, ($checksums -join "`n") + "`n", [Text.Encoding]::ASCII)
    Invoke-Adb push $tempChecksum "$BoardRoot/deployment.sha256" | Out-Null
} finally {
    if (Test-Path -LiteralPath $tempChecksum) { Remove-Item -LiteralPath $tempChecksum }
}
Invoke-Adb shell "cd $BoardRoot && sha256sum -c deployment.sha256 > deployment_verify.log && tail -n 3 deployment_verify.log"
$install = if ($SkipInstall) { '--skip-install' } else { '' }
Invoke-Adb shell "bash $BoardRoot/ladder/setup_board.sh $BoardRoot $install"
Write-Host "Deployment verified. Start the ladder from PC:"
Write-Host "adb -s $Serial shell `"cd $BoardRoot && bash -o pipefail -c 'bash ladder/run_ladder.sh 2>&1 | tee ladder.log'`""
Write-Host 'For App Lab, import a newly generated app ZIP; the host venv does not configure the App Lab container.'
