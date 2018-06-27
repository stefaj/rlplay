with import <nixpkgs> {};
with pkgs.python36Packages;

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
		    python36Full
                    cudatoolkit
                    cudnn
                    python36Packages.requests
                    python36Packages.pyglet
                    python36Packages.six
                    python36Packages.numpy
                    python36Packages.scipy
                   ];
   };

<<<<<<< HEAD
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
		    python36Full
                    cudatoolkit
                    cudnn
                    python36Packages.requests
                    python36Packages.pyglet
                    python36Packages.six
                    python36Packages.numpy
                    python36Packages.scipy
                   ];
   };

   l4optim = buildPythonPackage rec {
     pname = "l4optimizer";
     version = "1.0.0";
     name = "${pname}-${version}";

    src = (import <nixpkgs> {}).fetchFromGitHub {
      owner = "martius-lab/";
      repo = "l4-optimizer/";
      rev = "f3738535bdaf06b768306054b5dd6e6a654acdbc";
      sha256 = "1ycmhfbgr4gbh9hald0z9lyva7ryj3nj9z6g79ddqld0k1hhk3r9";
    };

    buildInputs = [ python36Packages.attrs python36Packages.six python36Packages.pytest
                    python36Packages.testtools python36Packages.joblib python36Packages.tqdm
                    python36Packages.scipy python36Packages.dill gym 
                    ];

    doCheck = false;

   };

=======
>>>>>>> 17c2e37a4c170bbfb68552039049b278dc0bcd60
in

buildPythonPackage{
    name = "numerai";
    buildInputs = [ 
                    ocl-icd
                    cudatoolkit
                    cudnn
                    python36Full
                    python36Packages.matplotlib
                    python36Packages.requests
                    python36Packages.pyglet
<<<<<<< HEAD
                    python36Packages.tensorflowWithCuda
                    python36Packages.setuptools
                    # python36Packages.pandas0.17.1
                    python36Packages.numpy
                    gym
                    l4optim
=======
                    # python36Packages.tensorflowWithCuda
                    python36Packages.setuptools
                    python36Packages.numpy
                    # python36Packages.pytorch
                    gym
>>>>>>> 17c2e37a4c170bbfb68552039049b278dc0bcd60
                    # python36Packages.Quandl
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

