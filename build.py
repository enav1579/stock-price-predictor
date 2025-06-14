import os
import sys
import shutil
import subprocess

def build_executable():
    # Create a spec file for PyInstaller
    spec_content = """
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=['streamlit', 'pandas', 'yfinance', 'plotly', 'sklearn', 'numpy'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
    name='StockPricePredictor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
    """
    
    # Write the spec file
    with open('StockPricePredictor.spec', 'w') as f:
        f.write(spec_content)
    
    # Install required packages
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'pyinstaller'])
    
    # Build the executable
    subprocess.run(['pyinstaller', 'StockPricePredictor.spec', '--clean'])
    
    print("Executable built successfully!")

if __name__ == "__main__":
    build_executable() 