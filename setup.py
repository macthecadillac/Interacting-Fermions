from setuptools import setup, find_packages
from subprocess import Popen
import shutil
import os
import sys
import stat


# key = module name (the exact name of the cargo project)
# val = destination of the generated shared object
rust_modules = {
    "triangular_lattice_ext": "hamiltonians",
}


def build_rust_binary(lib, dest_dir):
    os.chdir("rust/{}".format(lib))
    p = Popen(["cargo", "build", "--release"])
    p.wait()
    if p.returncode != 0:
        sys.exit("Cargo build failure. Abort.")
    os.chdir("../..")
    dest = "{}/{}.so".format(dest_dir, lib)
    shutil.copyfile("rust/{0}/target/release/lib{0}.so".format(lib),
                    dest)
    st = os.stat(dest)
    os.chmod(dest, st.st_mode | stat.S_IEXEC)


# build Rust projects
for lib, dest_dir in rust_modules.items():
    build_rust_binary(lib, dest_dir)


setup(
    name="spinsys",
    packages=find_packages(),
    install_requires=['numpy>=1.8', 'scipy>=0.13', 'cffi>=1.11'],
    include_package_data=True,
)
