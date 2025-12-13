# Running Face Recognition Service Locally on Windows 10

This guide explains how to run the face-recognition service directly on Windows 10 (outside of Docker) for development purposes.

## Prerequisites

- **Python 3.10** (specifically 3.10, not 3.11 or 3.12)
  - Download from: https://www.python.org/downloads/
  - The pre-compiled dlib wheel is built for Python 3.10

## Setup Instructions

### 1. Create a Virtual Environment

```powershell
cd c:\Projects\svema-services\face-recognition
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Create Windows-Specific Requirements File

Create a file named `requirements-windows.txt` with the following content:

```
flask==3.0.0
face-recognition==1.3.0
numpy==1.24.3
pillow==10.1.0
scikit-learn==1.3.2
psycopg2-binary==2.9.9
requests==2.31.0
./lib/dlib-19.22.99-cp310-cp310-win_amd64.whl
```

### 3. Install Dependencies

```powershell
pip install -r requirements-windows.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the `face-recognition` directory:

```env
FLASK_APP=app.py
FLASK_ENV=development
DATABASE_URL=postgresql://user:password@localhost:5432/svema
# Add other environment variables as needed
```

### 5. Run the Service

```powershell
python app.py
```

## Why Two Requirements Files?

- **`requirements.txt`**: Used by Docker (Linux containers) - builds dlib from source
- **`requirements-windows.txt`**: Used for local Windows development - uses pre-compiled dlib wheel

The pre-compiled wheel (`dlib-19.22.99-cp310-cp310-win_amd64.whl`) saves you from having to install Visual Studio Build Tools and CMake on Windows.

## Troubleshooting

### "dlib wheel is not compatible"
- Make sure you're using **Python 3.10** (not 3.11 or 3.12)
- Check: `python --version`

### Database Connection Issues
- Ensure PostgreSQL is running (either via Docker or locally)
- Update `DATABASE_URL` in your `.env` file to match your setup

### Missing Dependencies
- If you get import errors, try: `pip install --upgrade pip`
- Then reinstall: `pip install -r requirements-windows.txt`

## VS Code Debugging Setup

To debug the Flask app with breakpoints in VS Code:

### 1. Copy the launch.json template

Copy `launch.json.template` to your workspace `.vscode` folder:

```powershell
# Create .vscode folder if it doesn't exist
mkdir .vscode -ErrorAction SilentlyContinue

# Copy the template
copy face-recognition\launch.json.template .vscode\launch.json
```

### 2. Install Python extension

Make sure you have the **Python** extension installed in VS Code (by Microsoft).

### 3. Select Python interpreter

1. Press `Ctrl+Shift+P`
2. Type "Python: Select Interpreter"
3. Choose the interpreter from your virtual environment: `.\face-recognition\venv\Scripts\python.exe`

### 4. Start debugging

1. Set breakpoints in your Python code
2. Press `F5` or go to Run → Start Debugging
3. Select "Python: Flask Face Recognition" from the dropdown

The app will start on `http://localhost:5000` with the debugger attached.

### Environment Variables in launch.json

The `launch.json` includes these environment variables:
- `DATABASE_URL`: Connection string for PostgreSQL (adjust if needed)
- `FLASK_ENV`: Set to "development" for auto-reload
- `FLASK_DEBUG`: Enables Flask debug mode

You can customize these in `.vscode/launch.json` as needed.

## Recommended Workflow

For most development:
1. **Use Docker** for running the complete service stack (recommended)
2. **Use local Python with VS Code debugging** when you need to:
   - Debug Python code with breakpoints
   - Step through face recognition logic
   - Inspect variables and stack traces
   - Run unit tests locally
   - Develop new features with fast iteration

Docker ensures consistency with production and avoids Windows-specific issues.
