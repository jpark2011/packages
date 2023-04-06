#!/bin/bash
set -e

# option format: space<option>) # <comments>
usage() { echo "$0 usage:" && grep " .)\ #" $0; exit 0; }

DEVICE="cpu"
WORK_DIR=`pwd`
while getopts "hd:w:" opt; do
    case ${opt} in
        d) # device type: cpu, xpu; default is cpu
            DEVICE=$OPTARG
            ;;
        w) # working directory where pytorch packages are downloaded and built; default is current working directory where this script is called
            WORK_DIR=$OPTARG
            ;;
        h | *) # display help
            usage
            ;;
    esac
done

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

init() {
    if [ ! -d "$WORK_DIR" ]; then
        log "Error: Cannot find $WORK_DIR" 
        exit 1
    fi
    log "Packages will be downloaded under $WORK_DIR"

    # activate anaconda env
    anaconda_dir=$HOME/Kyle/anaconda3
    conda_env=pt-xpu
    source $anaconda_dir/etc/profile.d/conda.sh && conda activate $conda_env
}

build_cpu_pytorch() {
    pushd .
    cd $WORK_DIR
    # internal version
    pytorch_git=https://github.com/intel-innersource/frameworks.ai.pytorch.private-gpu
    pytorch_dir=frameworks.ai.pytorch.private-gpu
    if [ ! -d "$pytorch_dir" ]; then
        log "Cloning from $pytorch_git"
        git clone $pytorch_git
    fi
    cd $pytorch_dir
    git submodule sync && git submodule update --init --recursive
    pip install -r requirements.txt

    log "Building PyTorch"
    export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}

    python setup.py bdist_wheel
    python setup.py install

    unset CMAKE_PREFIX_PATH
    popd
}

build_xpu_pytorch() {
    pushd .
    cd $WORK_DIR
    # internal version
    pytorch_git=https://github.com/intel-innersource/frameworks.ai.pytorch.private-gpu
    pytorch_dir=frameworks.ai.pytorch.private-gpu
    if [ ! -d "$pytorch_dir" ]; then
        log "Cloning from $pytorch_git"
        git clone $pytorch_git
    fi
    cd $pytorch_dir
    git submodule sync && git submodule update --init --recursive
    pip install -r requirements.txt

    log "Building PyTorch"
    export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}

    python setup.py bdist_wheel
    python setup.py install

    unset CMAKE_PREFIX_PATH
    popd
}

build_xpu_ipex() {
    pushd .
    cd $WORK_DIR
    ipex_git=https://github.com/intel-innersource/frameworks.ai.pytorch.ipex-gpu.git
    ipex_dir=frameworks.ai.pytorch.ipex-gpu
    export USE_AOT_DEVLIST='pvc,ats-m75'
    # export USE_AOT_DEVLIST='ats-m75'
    # Enable ITT annotation in sycle kernel
    export USE_ITT_ANNOTATION=ON
    if [ ! -d "$ipex_dir" ]; then
        log "Cloning from $ipex_git"
        git clone $ipex_git
    fi
    cd $ipex_dir
    git submodule sync && git submodule update --init --recursive
    pip install -r requirements.txt
    DPCPP_ROOT=/opt/intel/oneapi/compiler/2023.0.0/
    source ${DPCPP_ROOT}/env/vars.sh 
    MKL_ROOT=/opt/intel/oneapi/mkl/2023.0.0
    source ${MKL_ROOT}/env/vars.sh
    log "Building IPEX"

    python setup.py bdist_wheel
    python setup.py install
    popd
}

init

if [ "$DEVICE" == "cpu" ]; then
    log "Building $DEVICE PyTorch"

    build_cpu_pytorch

elif [ "$DEVICE" == "xpu" ]; then
    log "Building $DEVICE PyTorch"

    # build_xpu_pytorch
    build_xpu_ipex

else
    log "Error: invalid device type $DEVICE"
    exit 1
fi
