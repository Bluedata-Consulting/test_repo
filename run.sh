set -e

uv venv edge --python 3.11

source edge/bin/activate

uv pip install -r requirements.txt

clear

python app.py
