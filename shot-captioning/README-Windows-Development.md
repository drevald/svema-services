# Running Shot Captioning Service Locally on Windows

This guide explains how to run the shot-captioning service directly on Windows (outside of Docker) for development purposes.

## Prerequisites

- **Python 3.10+** (3.10, 3.11, or 3.12)
  - Download from: https://www.python.org/downloads/
- **PostgreSQL database** accessible from your Windows host
  - Either running locally or via Docker
  - Default connection: `localhost:5433`
- **Sufficient RAM and disk space**
  - The BLIP2 model requires ~4-8GB RAM
  - Model files will be downloaded (~10GB) on first run

## Setup Instructions

### 1. Create a Virtual Environment

```powershell
cd c:\Projects\svema-services\shot-captioning
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

**Note**: On Windows, PyTorch installation may require specific commands. If you encounter issues:

```powershell
# For CPU-only (faster download, no GPU acceleration)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# For CUDA 11.8 (if you have an NVIDIA GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Then install remaining dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables

The service requires database connection settings. These are configured in VS Code's launch.json, but you can also create a `.env` file:

```env
FLASK_APP=app.py
FLASK_ENV=development
DB_HOST=localhost
DB_PORT=5433
DB_NAME=svema
DB_USER=postgres
DB_PASSWORD=postgres
MODEL_NAME=Salesforce/blip2-flan-t5-xl
```

### 4. Ensure Database is Running

The service needs access to the PostgreSQL database. If you're running it via Docker:

```powershell
# Check if postgres container is running
docker ps | grep postgres

# If not, start it (adjust container name as needed)
docker start svema-postgres-1
```

Or ensure your local PostgreSQL service is running on port 5433.

### 5. Run the Service

```powershell
python app.py
```

The service will:
1. Load the BLIP2 model (first run downloads ~10GB of model files)
2. Connect to the database
3. Start the Flask server on `http://localhost:5556`

## VS Code Debugging Setup

### 1. Select Python Interpreter

1. Press `Ctrl+Shift+P`
2. Type "Python: Select Interpreter"
3. Choose: `.\shot-captioning\.venv\Scripts\python.exe`

### 2. Start Debugging

1. Open the Run and Debug panel (`Ctrl+Shift+D`)
2. Select "**Shot Captioning Service**" from the dropdown
3. Press `F5` or click "Start Debugging"

The service will start on `http://localhost:5556` with the debugger attached.

### Environment Variables in launch.json

The launch configuration at `.vscode/launch.json` includes:
- `DB_HOST`, `DB_PORT`, `DB_NAME`: Database connection settings
- `DB_USER`, `DB_PASSWORD`: Database credentials (update as needed)
- `MODEL_NAME`: AI model to use (default: Salesforce/blip2-flan-t5-xl)

Update these in `.vscode/launch.json` to match your environment.

## API Endpoints

- `GET /health` - Health check
- `POST /caption` - Generate caption for uploaded image
- `POST /caption/shot/<shot_id>` - Generate caption for specific shot and save as comment
- `POST /harvest` - Process all shots without AI comments

## Troubleshooting

### Model Download Issues
- First run downloads ~10GB of model files from HuggingFace
- Ensure stable internet connection
- Files are cached in `~/.cache/huggingface/`

### Database Connection Issues
- Verify PostgreSQL is running: `docker ps` or check local service
- Test connection: `psql -h localhost -p 5433 -U postgres -d svema`
- Update credentials in `.vscode/launch.json` if needed

### Out of Memory Errors
- The BLIP2-flan-t5-xl model requires significant RAM
- Close other applications
- Consider using a smaller model (edit `MODEL_NAME` in launch.json)

### Import Errors
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

## Running Both Services Together

You can debug both services simultaneously:

1. Start "**Face Recognition Service**" (runs on port 5555)
2. Start "**Shot Captioning Service**" (runs on port 5556)

Both will run independently with full debugging support.

## Recommended Workflow

For development:
1. **Use Docker** for complete stack testing (recommended for integration)
2. **Use local Python with VS Code debugging** when you need to:
   - Debug Python code with breakpoints
   - Step through caption generation logic
   - Test model changes quickly
   - Inspect database interactions
   - Profile performance

## Performance Notes

- **First run**: Slow (model download + initialization ~5-10 minutes)
- **Subsequent runs**: Model loads from cache (~30-60 seconds)
- **GPU vs CPU**: GPU is ~10x faster for caption generation
  - CPU: ~2-5 seconds per image
  - GPU: ~0.2-0.5 seconds per image
