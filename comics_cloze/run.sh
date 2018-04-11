#!/bin/bash

# Usage
# ./run.sh models/model_name.py

export PYTHONPATH=/remote/bones/user/public/hima/sequences/comics:$PYTHONPATH

# To prevent file lock errors on the NFS
export HDF5_USE_FILE_LOCKING="FALSE"

FILE_TO_RUN=$1
shift # Shift in order to pass rest of args to the python script

python "${FILE_TO_RUN}" "$@"
