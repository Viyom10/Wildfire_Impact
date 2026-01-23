@echo off
REM Windows batch runner
REM Usage: run_experiment.bat [--config path\to\config.json] [additional args]

SETLOCAL ENABLEDELAYEDEXPANSION

SET CONFIG=configs\config_test.json

:parse_args
IF "%1"=="" GOTO args_done
IF "%1"=="--config" (
  SHIFT
  SET CONFIG=%1
  SHIFT
  GOTO parse_args
)
REM collect remaining args
SET EXTRA_ARGS=%EXTRA_ARGS% %1
SHIFT
GOTO parse_args

:args_done

REM Prefer venv python
IF EXIST venv\Scripts\python.exe (
  SET PYTHON=venv\Scripts\python.exe
) ELSE (
  WHERE python >NUL 2>&1
  IF ERRORLEVEL 1 (
    ECHO python not found. Please install Python 3 or create a virtualenv named 'venv'.
    EXIT /B 1
  ) ELSE (
    SET PYTHON=python
  )
)

ECHO Using %PYTHON%
ECHO Config: %CONFIG%
ECHO Extra args: %EXTRA_ARGS%

%PYTHON% run_experiment.py --config %CONFIG% %EXTRA_ARGS%
