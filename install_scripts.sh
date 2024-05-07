#!/bin/bash

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

export INST_DIR=$SCRIPTPATH/vef_venv

export BIN_DIR=$INST_DIR/bin

if [[ $1 == -f ]];then
    /bin/bash $SCRIPTPATH/install_env.sh -f $INST_DIR
else
    /bin/bash $SCRIPTPATH/install_env.sh $INST_DIR
fi

for FILE in scripts/vef_*.py
    do
    ln -sf "$SCRIPTPATH/$FILE" $BIN_DIR
    done
