with import <nixpkgs> {};
with pkgs.python27Packages;

let
   gym = buildPythonPackage rec {
     pname = "gym";
     version = "0.9.4";
     name = "${pname}-${version}";

     src = pkgs.fetchurl {
       url = "https://pypi.python.org/packages/f8/9f/b50f4c04a97e316ebfccae3104e5edbfe7bc1c687ee9ebeca6fa6343d197/gym-0.9.4.tar.gz";
       sha256 = "121qq4al4in5pmq4am8aa2g70476yp2kvk2bb0y29cdsj2kirycl";
     };

     doCheck = false;
     buildInputs = [
		    python27Full
                    cudatoolkit
                    cudnn
                    python27Packages.requests
                    python27Packages.pyglet
                    python27Packages.six
                    python27Packages.numpy
                    python27Packages.scipy
                   ];
   };

   tensorflow = buildPythonPackage rec {
     pname = "tensorflow";
     version = "1.4.0";
     name = "${pname}-${version}";

     src = pkgs.fetchurl {
       url = "https://pypi.python.org/packages/ec/47/b9621f12c2aaa892658382af9493e611a0f3ea5e3d7001709d2d31c65507/tensorflow_gpu-1.4.0rc0-cp27-cp27mu-manylinux1_x86_64.whl";
       sha256 = "121qq4al4in5pmq4am8aa2g70476yp2kvk2bb0y29cdsj2kirycl";
     };

     doCheck = false;
     buildInputs = [
		    python27Full
                    cudatoolkit
                    cudnn
                    python27Packages.requests
                    python27Packages.pyglet
                    python27Packages.six
                    python27Packages.numpy
                    python27Packages.scipy
                   ];
   };
in

buildPythonPackage{
    name = "numerai";
    buildInputs = [ 
                    ocl-icd
                    cudatoolkit
                    cudnn
                    python27Full
                    python27Packages.matplotlib
                    python27Packages.requests
                    python27Packages.pyglet
                    python27Packages.futures
                    python27Packages.tensorflowWithCuda
                    python27Packages.setuptools
                    # python27Packages.pandas0.17.1
                    python27Packages.numpy
                    gym
                    # python27Packages.Quandl
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

