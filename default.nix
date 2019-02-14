# default.nix
with import <nixpkgs> {};
stdenv.mkDerivation {
    name = "mpi_rust"; # Probably put a more meaningful name here
    buildInputs = [clang llvmPackages.libclang autoconf automake libtool gsl openmpi ];
    LIBCLANG_PATH = llvmPackages.libclang+"/lib";
}
