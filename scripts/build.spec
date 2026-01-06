# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for SCAT.
Build with: uv run pyinstaller scripts/build.spec
Output goes to: release/SCAT.exe
"""

import sys
from pathlib import Path

block_cipher = None

# Project root (one level up from scripts/)
PROJECT_ROOT = Path(SPECPATH).parent

a = Analysis(
    [str(PROJECT_ROOT / 'scripts' / 'launcher.py')],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=[
        (str(PROJECT_ROOT / 'scat' / 'resources' / 'icon.ico'), 'scat/resources'),
    ],
    hiddenimports=[
        'scat',
        'scat.main_gui',
        'scat.detector',
        'scat.features',
        'scat.classifier',
        'scat.analyzer',
        'scat.trainer',
        'scat.statistics',
        'scat.spatial',
        'scat.visualization',
        'scat.report',
        'scat.config',
        'scat.labeling_gui',
        'scat.cli',
        'sklearn',
        'sklearn.ensemble',
        'sklearn.ensemble._forest',
        'sklearn.preprocessing',
        'sklearn.model_selection',
        'sklearn.utils._typedefs',
        'sklearn.neighbors._partition_nodes',
        'scipy.stats',
        'scipy.ndimage',
        'scipy.spatial',
        'scipy.spatial.distance',
        'scipy.special._ufuncs_cxx',
        'scipy.linalg.cython_blas',
        'scipy.linalg.cython_lapack',
        'pandas',
        'numpy',
        'cv2',
        'PIL',
        'matplotlib',
        'matplotlib.backends.backend_agg',
        'seaborn',
        'jinja2',
        'PySide6',
        'PySide6.QtWidgets',
        'PySide6.QtCore',
        'PySide6.QtGui',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'PyQt5',
        'PyQt6',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='SCAT',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(PROJECT_ROOT / 'scat' / 'resources' / 'icon.ico'),
    distpath=str(PROJECT_ROOT / 'release'),
    workpath=str(PROJECT_ROOT / 'build'),
)
