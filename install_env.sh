#!/usr/bin/env bash

#Check if the script is running in a venv and deactivate it
INVENV=$(python3 -c 'import sys; print ("1" if hasattr(sys, "real_prefix") else "0")')
if [[ INVENV -eq 1 ]];then
    echo "exiting current venv"
    deactivate
fi

#The path to the dir where the script is located (IT MUST BE CENTERLINE PATH)
SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

error_activate() {
    echo "Could not activate the virtual environment $1"
    exit 1
}

check_venv() {
    PY_ENV=$(which python3)
    echo "Current python: $PY_ENV"

    #Check if the script is running in a venv and exit if not

    if [[ PY_ENV == *$1/bin/python3 ]];then
        echo "Not running in a venv. Abort."
        exit 1
    fi
}

build_venv () {
    echo "building $1"
    python3 -m venv $1

    source $1/bin/activate || error_activate $1
    check_venv

    echo "installing requirements"
    pip3 install --upgrade pip
    pip3 install wheel
    pip3 install -r "$SCRIPTPATH/requirements.txt"
    pip3 install -e $SCRIPTPATH
    deactivate
}

set_act_alias() {

    NAME=$(basename $1)

    if alias "act_$NAME" > /dev/null 2>&1
        then
            echo "Alias act_$1 already exists. The activate alias should be set manually."
        else
            case $SHELL in

                "/bin/zsh")
                    echo "alias act_$NAME='source $SCRIPTPATH/$1/bin/activate'" >> "$HOME/.zshrc"
                    ;;

                "/bin/bash")
                    echo "alias act_$NAME='source $SCRIPTPATH/$1/bin/activate'" >> "$HOME/.bashrc"
                    ;;

                *)
                    echo "Could not set alias to activate. Only zsh and bash are supported"
                    ;;

            esac
    fi

}


if [[ ( $# -eq 1 || ( $# -eq 2 && $1 == -f )) ]];then
    if [[ $# -gt 1 ]];then
        if [[ -d $2 ]];then
            rm -r $2
        fi
        build_venv $2
        set_act_alias $2

    elif [[ ! -d $1 ]];then
        build_venv $1
        set_act_alias $1

    else
        echo "Using already existing Virtual Environment $1. Use -f option to force rebuild from default values."
    fi

else
    echo "Bad usage. Options are install_env [-f] <venv> "
    exit 5
fi
exit 0
