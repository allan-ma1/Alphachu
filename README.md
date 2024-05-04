# APE-X Parallel Reinforcement Learning for research project use
First of all, I want to thank Lyusungwon for his concise implementation of Ape-X DQN in Pytorch framework for such an adorable Pikachu volleyball game and this repository copied from [his work](https://github.com/Lyusungwon/apex_dqn_pytorch). 

## Environment Settings
Use Ubuntu **22.04** with the same Python env settings. No code change is needed to make this repository compatible with your system.

## Required Packages
- **Wine** for initializing *.exe* file in Linux system;
- **Xvfb**, X virtual framebuffer for providing a Virtual Screen to display games and computer vision interface;
- **x11vnc** (optional), a VNC server, allows one to view remotely and interact with real X displays. Install this on the server.
- A **VNC viewer** (optional) software to visualize the pseudo screen for each Xvfb. Install this on your local host

Run the following lines in the terminal to install the required packages.
```bash
sudo apt-get update
sudo apt-get install wine winetricks
sudo apt-get install xvfb -y
```
For optional installations, please refer to abundant materials on Google. If you don't install the viewers, then there won't be any visuals, just outputs in the terminal.

## Python Environment 
I use Python version 3.7.16 for this repository by creating a new conda environment. If you want to do the same, run the command below if you have conda installed on your system and it will create a conda environment called **apex**.

```bash
conda create -n apex python==3.7.16
```
Install Pytorch dependencies from [Pytorch website](http://pytorch.org). For my implementation, both environments install the 1.3.0 version using:
```bash
conda install pytorch==1.13.0 torchvision==0.14.0 pytorch-cuda=11.7 -c pytorch -c nvidia
```
Install required python packages with specified packages version:
```bash
pip install -r requirements.txt
```
Currently, the contents of requirements.txt is as follows:
> opencv-contrib-python==4.8.1.78  \
> mss==7.0.1                       \
> numpy==1.21.6                    \
> tensorboardX==2.6.2.2            \
> pynput==1.3.10                  

## Start the Game
The code section below only needs to be run once to create the associated files and processes to start the game. 
Note: Every time you reboot the server or kill the process, you'll need to rerun the first line to initialize the process
- First line will use **Xvfb** to create the virtual Screen with accessible port number 1000 in pixels size 1280x1024 and 24 color channels. The last part says the output dir is path /dev/null that discards the output and
- Second line only needs to be run once for each port, and ```export DISPLAY=:1000"``` will be stored into ```.bashrc``` file forever.
```bash
Xvfb :1000 -ac -screen 0 1280x1024x24 > /dev/null &
echo "export DISPLAY=:1000" >> ~/.bashrc
````
To start the game, run this code section. This block will need to be run every time on seperate terminals.
- First line utilizes the **wine** to start the game ```pikachu.exe```
- Second line is used to start an x11vnc server session on the X11 display number 1000
```bash
DISPLAY=:1000 wine pika.exe
x11vnc -display :1000 # for port 1000
````

Notes:
- The part should be run N times if you want to create N actors in Ape-X structure. Increase the port number from 1000 to 1001 and so on incrementally for easy management.

Tips:

To identify list of running Xvfb processes, VM:
```
ps aux | grep Xvfb
```
Remember to kill each process window after training:
```
sudo kill -9 <pid>
```

## Start the Learner
```bash
python learner.py
```
Check the available input arguments in the learner.py.

## Start the actor
This process also need to be ran N times with corresponding increasing port number as you did in creating Virtual Screens.
Ensure the simnum matches the last few digits of the display port
```bash
DISPLAY=:1000 python actor.py --simnum 0
```
Ensure that you match simnum in actor with DISPLAY number in the following manner:
- simnum=0 --> DISPLAY:=1001
- simnum=1 --> DISPLAY:=1001
- etc.

Notes:
- The older important input --log_model is optional now. Actors will auto-select the latest model as the policy based on the timestamp.
