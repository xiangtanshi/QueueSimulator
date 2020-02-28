#  Intro of the repo(version2)  
  *components of the repo:  
  1. ciw module  
  2. queue simulator model code  
  3. experiments and results  
  4. current problems that needs to be solved  



__how to use the modified ciw module?__  
first create a new env(python version = 3.6) with anaconda, install pytorch and ciw, the replace the ciw module with the unzipped package in the repo.  

__what is the difference between the original ciw and the modified ciw?__  
I checked the whole ciw module and changed several parts of the codes in the original ciw module to promise that it can be applied to construct a computation graph directly using pytorch. Briefly speaking, most parameters of the ciw.create_network() can be updated through SGD and other algorithms that need to compute gradients of the model's parameters. I've run quite a few simulation of different network with different structure and parameters(both very simple network and complex network) and was sure that those changes I made will not influence the simulation process and final records of the ciw module. All the functions of ciw can be used just like before.  


__codes,expriment results__  
The codes(queue_simulator.py) and corresponding running results have already been uploaded, and it solves the problems of simulating   complex queue network with known structure and unknown continuous parameters.  
The codes(queue_simulator.py) utilizes the SmoothL1loss function and serves for 2 purposes:  
  1.figure out how to design well defined and meaningful features that can be applied to training
  2.observed the convergence condition of the loss and parameters and find out the regulation of training
for detailed explanation see '网络训练总结.pdf'  









