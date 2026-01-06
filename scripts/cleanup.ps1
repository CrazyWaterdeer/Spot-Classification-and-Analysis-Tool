# SCAT Project Cleanup Script
# Run from SCAT project root folder

Write-Host "=== SCAT Project Cleanup ===" -ForegroundColor Cyan

# 1. Create new folder structure
Write-Host "`n[1/6] Creating folder structure..." -ForegroundColor Yellow
$folders = @("data\images", "data\models", "data\results", "scripts", "release")
foreach ($folder in $folders) {
    if (!(Test-Path $folder)) {
        New-Item -ItemType Directory -Path $folder -Force | Out-Null
        Write-Host "  Created: $folder" -ForegroundColor Green
    } else {
        Write-Host "  Exists: $folder" -ForegroundColor Gray
    }
}

# 2. Move files
Write-Host "`n[2/6] Moving files..." -ForegroundColor Yellow

# Images folder
if (Test-Path "Images") {
    Move-Item -Path "Images\*" -Destination "data\images\" -Force -ErrorAction SilentlyContinue
    Remove-Item -Path "Images" -Force -Recurse -ErrorAction SilentlyContinue
    Write-Host "  Moved: Images/* -> data/images/" -ForegroundColor Green
}

# Results folder
if (Test-Path "results") {
    Move-Item -Path "results\*" -Destination "data\results\" -Force -ErrorAction SilentlyContinue
    Remove-Item -Path "results" -Force -Recurse -ErrorAction SilentlyContinue
    Write-Host "  Moved: results/* -> data/results/" -ForegroundColor Green
}

# Model file
if (Test-Path "model_rf.pkl") {
    Move-Item -Path "model_rf.pkl" -Destination "data\models\" -Force
    Write-Host "  Moved: model_rf.pkl -> data/models/" -ForegroundColor Green
}

# scat.spec to scripts/
if (Test-Path "scat.spec") {
    Move-Item -Path "scat.spec" -Destination "scripts\build.spec" -Force
    Write-Host "  Moved: scat.spec -> scripts/build.spec" -ForegroundColor Green
}

# Move dist/SCAT.exe to release/
if (Test-Path "dist\SCAT.exe") {
    Move-Item -Path "dist\SCAT.exe" -Destination "release\" -Force
    Write-Host "  Moved: dist/SCAT.exe -> release/" -ForegroundColor Green
}

# 3. Delete unnecessary files
Write-Host "`n[3/6] Removing unnecessary files..." -ForegroundColor Yellow

$deleteFiles = @("setup.py", "requirements.txt", "main.py")
foreach ($file in $deleteFiles) {
    if (Test-Path $file) {
        Remove-Item -Path $file -Force
        Write-Host "  Deleted: $file" -ForegroundColor Green
    }
}

# 4. Delete build artifacts
Write-Host "`n[4/6] Cleaning build artifacts..." -ForegroundColor Yellow

$deleteFolders = @("build", "dist", "scat.egg-info")
foreach ($folder in $deleteFolders) {
    if (Test-Path $folder) {
        Remove-Item -Path $folder -Force -Recurse
        Write-Host "  Deleted: $folder/" -ForegroundColor Green
    }
}

# 5. Delete __pycache__
Write-Host "`n[5/6] Cleaning Python cache..." -ForegroundColor Yellow
Get-ChildItem -Path . -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Write-Host "  Cleaned: __pycache__ folders" -ForegroundColor Green

# 6. Summary
Write-Host "`n[6/6] Cleanup complete!" -ForegroundColor Cyan
Write-Host "`nNew structure:" -ForegroundColor White
Write-Host @"
SCAT/
├── scat/                 # Source code
├── tests/                # Tests
├── scripts/              # Build scripts
│   └── build.spec
├── data/                 # User data (Git ignored)
│   ├── images/
│   ├── models/
│   └── results/
├── release/              # Built EXE (Git ignored)
├── .gitignore
├── LICENSE
├── README.md
├── pyproject.toml
└── uv.lock
"@

Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host "  1. Review the new .gitignore file"
Write-Host "  2. Build EXE: uv run pyinstaller scripts/build.spec"
Write-Host "  3. git add -A && git commit -m 'Reorganize project structure'"
