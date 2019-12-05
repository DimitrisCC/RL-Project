The current folder is the folder used for training.
The folder "agent" contains the files you need for testing our agents.
It contains the agent file, the model file, the utilities file and the .pth saved state dict model to be loaded.
The difference it that in the "agent" folder the files contain only what's needed for evaluation, without all the arguments needed to be pass in the main.py
That essentially means that the hyperparameters and the paths are hardcoded so that it simply runs without any contructor arguments or adjustments.