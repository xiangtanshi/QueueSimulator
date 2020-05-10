#  Intro of the repo(version2)  
  *components of the repo:  
  1. ciw module  
  2. queue simulator model code and wgan code 
  3. experiments and reports

__how to use the modified ciw module?__  
first create a new env(python version = 3.6) with anaconda, install pytorch and ciw, the replace the ciw module with the unzipped package in the repo.  

__what is the difference between the original ciw and the modified ciw?__  
I checked the whole ciw module and changed several parts of the codes in the original ciw module to promise that it can be applied to construct a computation graph directly using pytorch. Briefly speaking, most parameters of the ciw.create_network() can be updated through SGD and other algorithms that need to compute gradients of the model's parameters. I've run quite a few simulation of different network with different structure and parameters(both very simple network and complex network) and was sure that those changes I made will not influence the simulation process and final records of the ciw module. All the functions of ciw can be used just like before.  


__current codes,expriment results__  
/report/'关于排队系统结构和参数的实验和调研.pdf'
/code/complicated_NHPP.py
/code/ex_cycle_queue_sys.py








