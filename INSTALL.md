Prerequisites:
    - Internet connection
    - Provided competition data

- Install Anaconda python
    - Commands can be found here: https://conda.io/docs/user-guide/install/index.html
    - Linux users will run 'bash Anaconda-latest-Linux-x86_64.sh'
- Unzip project somewhere on your computer
- Navigate to the project root directory using a terminal
- Run 'mkdir data'
- Run 'mkdir data/raw'
- Navigate to the data/raw folder and unzip the contents of the provided zipped data file
- Run 'bash create-environment.sh' to create the environment
    - This assumes you are using the Anaconda Python distribution
    - This will create an environment named 'SemEval2019-4' and install all dependencies using enviornment.yml
- Run 'source activate SemEval2019-4'
- Run 'bash install-externals.sh'
- Run 'bash runit.sh'