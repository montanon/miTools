#!/bin/bash

# Exit on error, uninitialized variable, or error in pipeline
set -Eeo pipefail

env=tools

echo $MITOOLS

# Function to print errors in red and exit
print_error() {
    local last_command_exit_code=$?
    # ANSI color code for red
    RED='\033[0;31m'
    # No color (reset)
    NC='\033[0m'
    # Print the error message in red
    echo -e "${RED}ERROR: The command exited with status $last_command_exit_code.${NC}"
    echo -e "${RED}Error message: $1${NC}"
    # Exit with the error status
    exit $last_command_exit_code
}
trap 'print_error "$BASH_COMMAND"' ERR

# Function to print success messages in green
print_success() {
    local message=$1
    # ANSI color code for green
    GREEN='\033[0;32m'
    # No color (reset)
    NC='\033[0m'
    # Print the success message in green
    echo -e "${GREEN}SUCCESS: $message${NC}"
}

# Function to test Python module import
test_python_module() {
    local module_name=$1
    local test_command=${2-} # Use default empty string if not set
    if [[ -z "$test_command" ]]; then
        $PYTHON_PATH -c "import $module_name" && print_success "Successfully imported $module_name" || print_error "Testing of $module_name failed"
    else
        $PYTHON_PATH -c "$test_command" && print_success "Successfully tested $module_name" || print_error "Testing of $module_name failed"
    fi
}

source /opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh
if [[ "$CONDA_DEFAULT_ENV" == "$env" ]]; then
    echo "Deactivating $env environment."
    conda deactivate
else
    echo "The current: $CONDA_DEFAULT_ENV environment is not $env, so it won't be deactivated."
fi

# Remove the existing environment if it exists
conda env remove -n $env || true
rm -rf /opt/homebrew/Caskroom/miniconda/base/envs/$env
conda create -n $env python=3.8 -y

if [[ "$CONDA_DEFAULT_ENV" == "$env" ]]; then
    echo "The current: $CONDA_DEFAULT_ENV environment is $env, so it won't be activated."
else
    echo "Activating $env environment."
    conda activate $env
fi

PYTHON_PATH=$(which python)

conda install -c conda-forge seaborn numpy pandas matplotlib opencv pytorch torchvision jupyter ipywidgets jupyterlab_widgets openpyxl -y
test_python_module seaborn
test_python_module numpy
test_python_module pandas
test_python_module matplotlib
test_python_module cv2
test_python_module torch
test_python_module torchvision
test_python_module jupyter
test_python_module ipywidgets

conda install -c conda-forge tokenizers=0.11.3 -y
test_python_module tokenizers

$PYTHON_PATH -m pip install transformers
test_python_module transformers
$PYTHON_PATH -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('I love you'))"

conda install -c conda-forge scikit-learn nltk -y
test_python_module sklearn
test_python_module nltk

conda install -c conda-forge spacy -y
$PYTHON_PATH -m spacy download en_core_web_sm
$PYTHON_PATH -m spacy download es_core_news_sm
test_python_module spacy

conda install -c conda-forge fuzzywuzzy tqdm pdfminer pdfminer.six lxml -y
test_python_module fuzzywuzzy
test_python_module tqdm
test_python_module pdfminer
test_python_module lxml

conda install -c conda-forge ninja -y

$PYTHON_PATH -m pip install pybind11
test_python_module pybind11


original_path=$(pwd)
sudo rm -rf ~/.racplusplus/RACplusplus
sudo rm -rf ~/.racplusplus/RACplusplus/_skbuild
sudo rm -rf ~/.racplusplus && mkdir ~/.racplusplus && cd ~/.racplusplus
git clone git@github.com:porterehunley/RACplusplus.git
cd RACplusplus

# Commented out: `#$PYTHON_PATH -m pip install pybind11` from ./dependencies_mac.sh
# Avoid package managing conflict errors by `sudo pip`
sed -i.bak '/$PYTHON_PATH -m pip install pybind11/s/^/#/' ./dependencies_mac.sh

sudo chmod +x ./dependencies_mac.sh
sudo ./dependencies_mac.sh

PYTHON_PATH=$(which python)
$PYTHON_PATH -m pip install scikit-build
test_python_module skbuild

$PYTHON_PATH setup.py install
mkdir build && cd build && sudo rm -rf * # Clean out the build directory
conda install -c conda-forge ninja
NINJA=$(which ninja)
CC=$(which gcc)
CXX=$(which g++)
cmake -G Ninja -DCMAKE_MAKE_PROGRAM=$(NINJA) -DCMAKE_C_COMPILER=$(CC) -DCMAKE_CXX_COMPILER=$(CXX) ..
ninja
cd ..
echo PWD:$(pwd)
$PYTHON_PATH -m pip install .
test_python_module racplusplus
cd "$original_path"

$PYTHON_PATH -m pip install xlsxwriter country_converter pycountry

conda install -c conda-forge numba -y
test_python_module numba

conda install -c conda-forge umap-learn -y
test_python_module umap-learn

conda install -c conda-forge Unidecode -y
test_python_module unidecode

python -m pip install countryinfo -y
test_python_module countryinfo

cd $MITOOLS
PYTHON_PATH -m pip install -e .
test_python_module mitools
cd "$original_path"

conda install -c conda-forge ploomber -y
test_python_module ploomber

ipython kernel install --user --name=$env
