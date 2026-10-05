@echo off
REM Run this in a SECOND terminal window while benchmark.py is running.
REM It shows real CUDA compute utilization every 1 second.
REM This is what Task Manager hides by default.
REM
REM You will see something like:
REM   # gpu    sm   mem   enc   dec   jpg   ofa
REM   # Idx     %     %     %     %     %     %
REM       0    94    52     0     0     0     0
REM
REM "sm" column is shader/compute utilization. 90+ means GPU is maxed out.
REM Task Manager shows the "3D" engine, not the "sm" compute engine.

nvidia-smi dmon -s u -d 1
