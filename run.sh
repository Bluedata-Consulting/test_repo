set -e

uv venv edge

source edge/bin/activate

uv pip install -r requirements.txt

clear

python app.py