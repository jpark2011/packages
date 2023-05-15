#!/bin/bash
set -e

# option format: space<option>) # <comments>
usage() { echo "$0 usage:" && grep " .)\ #" $0 && echo "Example: $0 -v 2.0.1 -w /home/intel/work/pkg_build/libtorch-2.0"; exit 0; }

MKL=1
TBB=0
WORK_DIR=`pwd`
PYTHON_VER=3.9
# libtorch version
PYTORCH_VER="main"
while getopts "hatv:w:" opt; do
    case ${opt} in
        a) # Try to use ATen library without MKL; default is to build with MKL enabled. Should build where MKL cannot be found by CMake: ex) libtorch_nomkl venv
            MKL=0
            ;;
        t) # Use TBB instead of OpenMP.
            TBB=1
            ;;
        v) # PyTorch version to specify. Default is main
            PYTORCH_VER=$OPTARG
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

create_conda_env() {
    if [ "$1" == "libtorch_nomkl" ]; then
        conda create -y -n $1 python=$PYTHON_VER && conda activate $1
        conda install -y numpy ninja cmake
        conda install -y -c conda-forge libstdcxx-ng=12
    else
        log "Error: cannot create $conda_env venv"
        exit 1
    fi
}

init() {
    if [ ! -d "$WORK_DIR" ]; then
        log "Error: Cannot find $WORK_DIR" 
        exit 1
    fi
    log "Packages will be downloaded under $WORK_DIR"

    # activate anaconda env
    # To Do: To check/manage env automatically
    anaconda_dir=$HOME/anaconda3
    if [ $MKL == 1 ]; then
        conda_env=libtorch_mkl
    else
        conda_env=libtorch_nomkl
    fi
    # Should stop here if required venv is not found
    source $anaconda_dir/etc/profile.d/conda.sh 
    if { conda env list | grep "$conda_env"; } >/dev/null 2>&1; then
        log "Activated venv $conda_env"
        conda activate $conda_env
    else
        log "Creating a venv $conda_env..."
        create_conda_env $conda_env
    fi
}

# MKL enabled libtorch
build_mkl_libtorch() {
    pushd .
    cd $WORK_DIR
    # public version
    pytorch_git=https://github.com/pytorch/pytorch.git
    # You can modify as necesseary
    pytorch_dir=pytorch
    # internal version
    # pytorch_git=https://github.com/intel-innersource/frameworks.ai.pytorch.private-gpu
    # pytorch_dir=frameworks.ai.pytorch.private-gpu
    if [ ! -d "$pytorch_dir" ]; then
        log "Cloning from $pytorch_git to $WORK_DIR/$pytorch_dir"
        git clone $pytorch_git $pytorch_dir
    fi
    cd $pytorch_dir
    git checkout ${PYTORCH_VER}
    git submodule sync && git submodule update --init --recursive
    pip install -r requirements.txt

    log "Building PyTorch"

    if [ $TBB == 0 ]; then
        libtorch_dir=libtorch-${PYTORCH_VER}-mkl
    else
        libtorch_dir=libtorch-${PYTORCH_VER}-mkl-tbb
    fi

    rm -rf ../${libtorch_dir}-build
    mkdir ../${libtorch_dir}-build
    cd ../${libtorch_dir}-build
    export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}

    if [ $TBB == 0 ]; then
        cmake -DBUILD_SHARED_LIBS:BOOL=ON -DCMAKE_BUILD_TYPE:STRING=Release -DCMAKE_INSTALL_PREFIX:PATH=../${libtorch_dir} \
        -DUSE_CUDA:BOOL=OFF -DUSE_MPI:BOOL=OFF -DUSE_KINETO:BOOL=OFF -DBUILD_BINARY:BOOL=OFF \
        -DUSE_MKL:BOOL=OFF -DUSE_BLAS=0 -DUSE_MKLDNN=ON -DUSE_OPENMP=ON -DUSE_TBB=OFF \
        -DGLIBCXX_USE_CXX11_ABI=0 \
        ../${pytorch_dir} 2>&1 | tee -a ./cmake.log
    else
        cmake -DBUILD_SHARED_LIBS:BOOL=ON -DCMAKE_BUILD_TYPE:STRING=Release -DCMAKE_INSTALL_PREFIX:PATH=../${libtorch_dir} \
        -DUSE_CUDA:BOOL=OFF -DUSE_MPI:BOOL=OFF -DUSE_KINETO:BOOL=OFF -DBUILD_BINARY:BOOL=OFF \
        -DUSE_TBB:BOOL=ON -DUSE_SYSTEM_TBB:BOOL=ON -DCMAKE_PREFIX_PATH=/opt/intel/oneapi/tbb/latest \
        -DATEN_THREADING=TBB -DMKL_THREADING=TBB -DBLAS=MKL -DMKLDNN_CPU_RUNTIME=TBB -DUSE_MKLDNN=1 \
        -DGLIBCXX_USE_CXX11_ABI=0 \
        ../${pytorch_dir} 2>&1 | tee -a ./cmake.log
    fi
    
    cmake --build . --target install --config Release -- -j 2>&1 | tee ./build.log
    unset CMAKE_PREFIX_PATH

    # build-hash, build-version
    cd ../${libtorch_dir}
    echo "$(pushd ../${pytorch_dir} && git rev-parse HEAD)" > ./build-hash
    # ../${pytorch_dir}/torch/csrc/api/include/torch/version.h 
    echo "${PYTORCH_VER}" | sed -r 's/v//g' > ./build-version

    popd
}

build_cpu_ipexlibtorch() {
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
    MKL_ROOT=/opt/intel/oneapi/mkl/2023.0.0
    source ${MKL_ROOT}/env/vars.sh
    log "Building IPEX"

    libtorch_dir=libtorch-2.1.0-nomkl
    LIBTORCH_PATH=../${libtorch_dir} python setup.py bdist_cppsdk
    popd
}

build_aten_libtorch() {
    pushd .
    cd $WORK_DIR
    # internal version
    pytorch_git=https://github.com/intel-innersource/frameworks.ai.pytorch.private-gpu
    # pytorch_dir=frameworks.ai.pytorch.private-gpu
    pytorch_dir=pytorch
    if [ ! -d "$pytorch_dir" ]; then
        log "Cloning from $pytorch_git"
        git clone $pytorch_git $pytorch_dir
    fi
    cd $pytorch_dir
    git submodule sync && git submodule update --init --recursive
    pip install -r requirements.txt

    log "Building PyTorch"

    libtorch_dir=libtorch-2.1.0-nomkl
    rm -rf ../${libtorch_dir}-build
    mkdir ../${libtorch_dir}-build
    cd ../${libtorch_dir}-build
    export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}

    # conda uninstall mkl
    # sudo mv /opt/intel/oneapi/mkl /opt/intel/oneapi/mkl_tmp
    # CMake Warning:
    #  Manually-specified variables were not used by the project:
    #  USE_MKL
    cmake -DBUILD_SHARED_LIBS:BOOL=ON -DCMAKE_BUILD_TYPE:STRING=Release -DCMAKE_INSTALL_PREFIX:PATH=../${libtorch_dir} \
    -DUSE_CUDA:BOOL=OFF -DUSE_MPI:BOOL=OFF -DUSE_KINETO:BOOL=OFF -DBUILD_BINARY:BOOL=OFF \
    -DUSE_MKL:BOOL=OFF -DUSE_BLAS=0 -DUSE_MKLDNN=ON -DUSE_OPENMP=ON -DUSE_TBB=OFF \
    ../${pytorch_dir} 2>&1 | tee -a ./cmake.log
    # sudo mv /opt/intel/oneapi/mkl_tmp /opt/intel/oneapi/mkl
    
    cmake --build . --target install --config Release -- -j 2>&1 | tee ./build.log

    unset CMAKE_PREFIX_PATH

    # build-hash, build-version
    cd ../${libtorch_dir}
    echo "$(pushd ../${pytorch_dir} && git rev-parse HEAD)" > ./build-hash
    # ../${pytorch_dir}/torch/csrc/api/include/torch/version.h 
    echo "2.1.0" > ./build-version

    popd
}

init

if [ $MKL == 1 ]; then
    log "Building MKL enabled libtorch"

    build_mkl_libtorch

else
    log "Building ATen based PyTorch"

    build_aten_pytorch
fi
