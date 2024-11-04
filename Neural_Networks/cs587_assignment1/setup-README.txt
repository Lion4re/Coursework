Setup your Python environment to run the assignment code using PYTHON 3.10 !

===== [Option 1] Use Anaconda =====

The preferred approach for installing all the assignment dependencies is to use Anaconda, 
which is a Python distribution that includes many of the most popular Python packages for 
science, math, engineering and data analysis. 
Once you install it you can skip all mentions of requirements and you're ready to go directly to working on the assignment.

You need to work using Python 3.10 for this assignment (and, most likely, the rest assignments too).
So, using Anaconda environment (whatever version 4.x), it is easy to create an environment to use with Python3.10.

[Automatic environment setup using file]
(the environment.yml file is located in the assignment1 folder)
>> cd cs587_assignment1
>> conda env create --name <environment_name> --file environment.yml

OR

[Manual environment setup from the terminal]
Two steps: 
1. Create the working environment.
>> cd cs587_assignment1
>> conda create --name <environment_name> python=3.10
>> conda activate <environment_name>
>> jupyter notebook

2. Install the required packages 
    Method A (all at once):
>> conda install numpy pandas matplotlib scikit-learn scikit-image tqdm
If you want Pytorch to run on GPU (and you have one available) run a custom command from https://pytorch.org/
To install Pytorch that runs only on CPU, run:
>> conda install pytorch torchvision torchaudio cpuonly -c pytorch
>> conda install jupyter

    Method B (if A has an issue):
Install only the necessary:
1. numpy
2. matplotlib
and whatever else pops up as uninstalled (do not rely on requirements.txt as there might be an issue with your machine and the version that we request). Anaconda will automatically find the compatible and latest versions!


===== [Option 2] =====
Manual installation, virtual environment using Python 3.10 distro installed in your machine:

If you'd like to (instead of Anaconda) go with a more manual and risky installation route you will likely want to create a virtual environment for the project.
If you choose not to use a virtual environment, it is up to you to make sure that all dependencies for the code are installed globally on your machine. 

To set up a virtual environment, run the following (the requirements.txt file in located in the assignment1 folder).

cd assignment1
sudo pip install virtualenv      # This may already be installed
virtualenv .env                  # Create a virtual environment
source .env/bin/activate         # Activate the virtual environment
pip3 install numpy pandas matplotlib scikit-learn scikit-image tqdm
pip3 install torch torchvision torchaudio
# Work on the assignment for a while ...
deactivate                       # Exit the virtual environment

The following link might be helpful http://stackoverflow.com/questions/5506110/is-it-possible-to-install-another-version-of-python-to-virtualenv
