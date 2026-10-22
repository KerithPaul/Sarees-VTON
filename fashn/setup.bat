@echo off
echo ============================================
echo   Saree Try-On Setup
echo ============================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.10+ from python.org
    pause
    exit /b 1
)

echo Creating virtual environment...
python -m venv venv
call venv\Scripts\activate

echo.
echo Installing packages...
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo ============================================
echo   Setup complete!
echo ============================================
echo.
echo Next steps:
echo   1. Add your FASHN API key to .env  (copy .env.example to .env)
echo   2. Add model pose photos to model_poses\  (front.jpg, side.jpg, back.jpg)
echo      OR use Auto-Generate mode which needs no pose photos
echo   3. Run the app:
echo        venv\Scripts\activate
echo        python app.py
echo.
echo The app will open in your browser at http://localhost:7860
echo.
pause
