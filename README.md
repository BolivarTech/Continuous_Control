# Project Details

This project implements one agent that interact with a Unity simulated
environment where it needs to control a double joined arm to keep arm's
end into one moving sphere volume.

The agent needs to have an average score of 30 or more over 100 episodes
to consider the environment solved.

# Getting Started

The Continuous_Control to run needs some dependencies to be installed
before.

Because it uses the Unity Engine to run the environment simulator is
necessary to install it and the ML-Agents toolkit.

Clone the project's repository local on your machine

> git clone https://github.com/BolivarTech/Continuous_Control.git
>
> cd Continuous_Control

Run the virtual environment on the console

## Windows:

> .\venv\Scripts\activate.bat

## Linux:

> source ./venv/Scripts/activate

Install the following modules on python.

-   Numpy (v1.19.5+)

-   torch (v1.8.2)

-   tensorflow (v1.7.1)

-   mlagents (v0.27.0)

-   unityagents

-   protobuf (v3.5.2)

To install it, on the projects root you need to run on one console

> pip install .

# Instructions

To run the agent you first need to activate the virtual environment.

## Windows:

> .\venv\Scripts\activate.bat

## Linux:

> source ./venv/Scripts/activate

On the repository's root you can run the agent to get a commands'
description

usage: main.py [-h] [-t] [-p] [-a ACTOR] [-c CRITIC]

optional arguments:

-h, --help show this help message and exit

-t, --train Perform a model training, if -a and -c are not specified a
new model is trained.

-p, --play Perform the model playing.

-a ACTOR, --actor ACTOR Path to a pytorch actor model file.

-c CRITIC, --critic CRITIC Path to a pytorch critic model file.

The -t option is used to train a new model or if -a and -c option are
selected continues training the selected model.

The -p option is used to play the agent on the environment using the
model specified on the -a flag, the -c flag is not needed.

The -h option shows the command's flags help
