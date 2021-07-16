# SPEAR for ULTRON

This is an add-on to the SPEAR tool (https://github.com/quasylab/spear), related to project ULTRON.

SPEAR is a simple Python tool that permits estimating the distance between two systems.

Each system consists of three distinct components:

    a process describing the behaviour of the program;
    a data space;
    an environment evolution describing the effect of the environment on the data space.

Two systems are compared via a (hemi)metric, called the evolution metric. Thanks to the possibility of extrapolating process behaviour from that of the system typical of our model, this metric allows us to

    verify how well a program is fulfilling its tasks by comparing it with its specification;
    compare the activity of different programs in the same environment;
    compare the behaviour of one program with respect to different environments and changes in the initial environmental conditions.

We have then introduced a probabilistic temporal logic, called Evolution Temporal Logic (EvTL), that allows us to express requirements on the probability distributions describing the transient behaviour of programs, in the presence of uncertainties.
EvTL is equipped with a robustness semantics that is evaluated by means of the evolution metric.
In this add-on we have developed a statistical model checking algorithm for EvTL specifications.

In tanks.py we show how the updated SPEAR can be used to model a variant of the three-tanks laboratory experiment. 
There are three identical tanks connected by two pipes. Water enters in the first and in the last tank by means of a pump and an incoming pipe, respectively. The last tank is equipped with an outlet pump. We assume that water flows through the incoming pipe with a rate that is determined by the environment, whereas the flow rate through the two pumps is under the control of a software component. 
The task of the system consists in guaranteeing that the levels of water in the three tanks fulfil some given requirements.

Moreover, we also provide a simple example of the evaluation of the robustness of the considered system with respect to two different EvTL formulae, in different scenarios, i.e., in the presence of various sources of perturbations and uncertainties.

In tanks_error.py we provide a simple tool for the estimation of the error that we commit in the statistical approximation of the behaviour of the system.


# Download

To download SPEAR you have just to clone GitHub project:

git clone https://github.com/gitUltron/Ultron.git

Run this command in the folder where you want to download the tool.
How to run experiments

To run experiments Python3 is needed. Moreover, the following Python packages must be available:

    numpy >= 1.18.4
    scipy >= 1.4.1
    matplotlib >= 3.0.3

To install all the required packages you can just run:

    pip install -r requirements.txt
