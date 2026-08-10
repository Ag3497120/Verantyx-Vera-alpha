# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['vera_entry.py'],
    pathex=[],
    binaries=[],
    # The failure-domain packs are DATA now, not code. Without this the
    # frozen binary would load two built-in packs and silently lose the
    # twelve JSON ones — a failure that looks like 'the field packs were
    # never written' rather than 'they were not shipped'.
    datas=[('verantyx/failure_packs', 'verantyx/failure_packs'),
           # The Japanese grammar is data, not code — a frozen binary
           # without it would run with no Japanese at all, silently.
           ('verantyx/lang_data', 'verantyx/lang_data')],
    hiddenimports=['mcp', 'mcp.server.fastmcp'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='vera-memory',
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
