if [ -z "$1" ]; then
   env_name=nebfir
else
   env_name=$1
fi

# Create an *anaconda3* virtual environment: (Assuming user is in base directory ```NebFIR/``` )
cd virtual_env
source virtual_env.sh $env_name


# Compile the Cython scripts (.pyx) so they can be called and ran by other scripts: (Assuming user is in base directory ```NebFIR/``` )
python setup.py develop build_ext --inplace


# Create the ```PATHS.json``` file containing usefull paths to the dataset, lists, etc. : (Assuming user is in base directory ```NebFIR/``` )
python -c 'from nebfir.tools.tools_path import create_paths; create_paths()'

