@echo off
REM Create datasets directory if it doesn't exist
if not exist datasets (
    mkdir datasets
)
cd datasets

REM Download CEPDOF.zip
powershell -Command "Invoke-WebRequest -Uri https://www.bu.edu/vip/files/CEPDOF.zip -OutFile CEPDOF.zip"
REM Unzip CEPDOF.zip
powershell -Command "Expand-Archive -Path CEPDOF.zip -DestinationPath ."

REM Download WEPDTOF.zip
powershell -Command "Invoke-WebRequest -Uri https://www.bu.edu/vip/files/WEPDTOF.zip -OutFile WEPDTOF.zip"
REM Unzip WEPDTOF.zip
powershell -Command "Expand-Archive -Path WEPDTOF.zip -DestinationPath ."

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
