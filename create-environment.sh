read -p "Create project environment (y/n)?" CONT

if [ "$CONT" == "n" ]; then
  echo "exit";
else

conda env create --name 'SemEval2019-4' -f 'environment.yml'
fi