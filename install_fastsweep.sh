git clone https://github.com/rgl-epfl/fastsweep.git
cp distance_marcher.cpp fastsweep/src
cd fastsweep
git submodule update --init --recursive
python3 setup.py install
cd ..
