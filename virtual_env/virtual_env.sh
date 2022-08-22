## To get packages in txt file run:  conda env export --no-builds > virtual_env/environment.yml

## RUN: source virtual_env.sh
ENVS=$(conda env list)


if [ -z "$1" ]; then
   env_name=nebfir
else
   env_name=$1
fi
echo "Environment name: $env_name"


if [[ $ENVS = *"$env_name"* ]]; then
   echo "$env_name env already exists..."
else 
   conda env create -f environment.yml -n $env_name
fi;

conda activate $env_name

cd ..
