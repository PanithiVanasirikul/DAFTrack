@echo off
REM Create datasets directory if it doesn't exist
if not exist datasets (
    mkdir datasets
)
cd datasets

REM Remove __MACOSX directories
for /d /r . %%d in (__MACOSX) do (
    if exist "%%d" (
        rmdir /s /q "%%d"
    )
)

REM Remove .DS_Store files
for /r . %%f in (.DS_Store) do (
    del /f /q "%%f"
)

cd ..
