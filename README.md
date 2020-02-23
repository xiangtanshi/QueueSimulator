#  Intro of the repo(version 1)  
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
the codes I ran in the past few days have already been uploaded, and it still requires further fix. The expriment results are waiting to be updated  

__problems__  
see pdf in the repo







