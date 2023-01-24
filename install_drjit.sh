git clone https://github.com/mitsuba-renderer/drjit.git
cd drjit
git submodule update --init --recursive
pip3 install scikit-build
pip3 install pybind11
python3 setup.py install
cd ..
