#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

pick_python() {
  for py in python3.12 python3.11 python3.10; do
    if command -v "${py}" >/dev/null 2>&1; then
      echo "${py}"
      return 0
    fi
  done
  return 1
}

if ! PYTHON_BIN="$(pick_python)"; then
  echo "No compatible Python found for TensorFlow."
  echo "Install Python 3.11 or 3.12, then rerun setup."
  echo "macOS (Homebrew): brew install python@3.11"
  exit 1
fi

echo "Using ${PYTHON_BIN} for virtual environment"
echo "Creating virtual environment at ${VENV_DIR}"
"${PYTHON_BIN}" -m venv --clear "${VENV_DIR}"

echo "Activating virtual environment"
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

echo "Upgrading pip"
python -m pip install --upgrade pip

echo "Installing dependencies from requirements.txt"
pip install -r "${ROOT_DIR}/requirements.txt"

echo
echo "Setup complete."
echo "Run the app with:"
echo "source .venv/bin/activate && streamlit run app.py"
