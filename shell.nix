with import <nixpkgs> {};
with pkgs.python27Packages;

buildPythonPackage{
    name = "numerai";
    buildInputs = [ 
                    ocl-icd
                    clblas-cuda
                    cudatoolkit
                    cudnn
                    python27Full
                    python27Packages.matplotlib
                    python27Packages.futures
                    python27Packages.future
                    python27Packages.twisted
                    python27Packages.scipy
                    python27Packages.Keras
                    python27Packages.tensorflow
                    # python27Packages.TheanoWithCuda
                    python27Packages.h5py
                    python27Packages.setuptools
                    python27Packages.pandas
                    python27Packages.httplib2
                    python27Packages.urllib3
                    python27Packages.joblib
                    python27Packages.numpy
                    python27Packages.websocket_client
                    # python27Packages.libgpuarray-cuda
                    # clblas-cuda
                    # opencl-headers
                    # ocl-icd
                   ]; 
  shellHook = ''
  # set SOURCE_DATE_EPOCH so that we can use python wheels
  SOURCE_DATE_EPOCH=$(date +%s)
  CPATH=$CPATH:~/.local/include
  LIBRARY_PATH=$LIBRARY_PATH:~/.local/lib
  LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.local/lib

  # LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/nix/store/wllbqw6zg5z355fzb6adblyd44jlbxv5-nvidia-x11-375.66/lib

  # :/nix/store/x13964hyy5pfkz7x6g9bhbdnnbz9ncgq-nvidia-libs/lib

  '';

}

