#!/usr/bin/env bash

set -euo pipefail

cd $(dirname $0)

# Create a virtual environment to run our code
VENV_NAME="${VIAM_MODULE_DATA}/venv"
PYTHON="$VENV_NAME/bin/python"

# uncomment for hot reloading support
# export PATH=$PATH:$HOME/.local/bin
# source $VENV_NAME/bin/activate
# uv pip install ./dist/*.whl -q

# Be sure to use `exec` so that termination signals reach the python process,
# or handle forwarding termination signals manually
echo "Starting module..."
exec $PYTHON -m main $@
