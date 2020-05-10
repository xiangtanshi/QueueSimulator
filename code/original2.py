import ciw
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import autograd
import numpy as np
import matplotlib.pyplot as plt

class Simunet(nn.Module):
    def __init__(self):
        super(Simunet,self).__init__()
        self.params1 = nn.Parameter(torch.Tensor(2).uniform_(0.3,1.3))    #arrival params
        self.params2 = nn.Parameter(torch.Tensor(3).uniform_(0.3,1.3))    #service params


    def seq_process(self,seq):
        length = len(seq)
        if length<50:
            result = [0 for _ in range(50)]
            result[0:length] = seq
        else:
            result = seq[0:50]
        return result


    def recs_parser(self,recs, node_num):
        '''
        take collected object records from ciw's simulation and sort out the data in each node
        :param recs:
        :return: the list containing statistical info of each node's data
        '''

        parser_data = torch.zeros(node_num,3*50)  #feature length = node_num * 3*50

        for i in range(1, node_num + 1):

                node_arri = [r.arrival_date for r in recs if r.node == i and 10<r.arrival_date<100]     #10 for warm up
                node_arri.sort()
                for j in range(len(node_arri)):
                    node_arri[j] = node_arri[j] - node_arri[0]
                arri = self.seq_process(node_arri)

                node_arri_ = [r.arrival_date for r in recs if r.node == i and 200>r.arrival_date>110]       #10 for cool down
                node_arri_.sort()
                for j in range(len(node_arri_)):
                    node_arri_[j] = node_arri_[j] - node_arri_[0]
                arri_ = self.seq_process(node_arri_)

                node_ser = [r.service_time for r in recs if r.node == i ]
                node_ser.sort()
                ser = self.seq_process(node_ser)

                cat_data = arri + arri_ + ser

                for k in range(3*50):
                    parser_data[i-1][k] = cat_data[k]

        return parser_data


    def forward(self,seed):

        N = ciw.create_network(
            arrival_distributions=[ciw.dists.Exponential(self.params1[0]),
                                   ciw.dists.Exponential(self.params1[1]),
                                   ciw.dists.NoArrivals()],
            service_distributions=[ciw.dists.Normal(self.params2[0],0.5),
                                   ciw.dists.Exponential(self.params2[1]),
                                   ciw.dists.Exponential(self.params2[2])],
            routing=[[0.0, 0.3, 0.7],
                     [0.0, 0.0, 1.0],
                     [0.0, 0.0, 0.0]],
            number_of_servers=[2,3,4]
        )
        #simulate 25 times, 25 is the batchsize
        batch = torch.zeros(10,3,150)
        for i in range(10):
            ciw.seed(seed+i)
            Q = ciw.Simulation(N)
            Q.simulate_until_max_time(220)
            recs = Q.get_all_records()
            batch[i] = self.recs_parser(recs,node_num=3)
        return batch

    def Realnet(self,seed):

        N = ciw.create_network(
            arrival_distributions=[ciw.dists.Exponential(0.6),
                                   ciw.dists.Exponential(0.3),
                                   ciw.dists.NoArrivals()],
            service_distributions=[ciw.dists.Normal(1.5,0.5),
                                   ciw.dists.Exponential(0.25),
                                   ciw.dists.Exponential(0.8)],
            routing=[[0.0, 0.25, 0.75],
                     [0.0, 0.0, 1.0],
                     [0.0, 0.0, 0.0]],
            number_of_servers=[2,2,3]
        )
        batch = torch.zeros(10, 3, 150)
        for i in range(10):
            ciw.seed(seed + i)
            Q = ciw.Simulation(N)
            Q.simulate_until_max_time(220)
            recs = Q.get_all_records()
            batch[i] = self.recs_parser(recs, node_num=3)
        return batch

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.main_module = nn.Sequential(
            #state=10*3*150
            nn.Conv1d(in_channels=3,out_channels=6, kernel_size=5,stride=3,padding=1),
            nn.ReLU(True),
            #state=10*6*50
            nn.Conv1d(in_channels=6, out_channels=12, kernel_size=5, stride=3, padding=1),
            nn.ReLU(True),
            #state=10*12*16
            nn.Conv1d(in_channels=12, out_channels=24, kernel_size=16, stride=1, padding=0),
            #state=10*24*1
        )

    def forward(self,x):
        return self.main_module(x)


class Simulator(object):
    def __init__(self):
        self.model = Simunet()
        self.D = Discriminator()
        if torch.cuda.is_available():
            self.cuda = True
            self.D = self.D.cuda()

        self.learning_rate_1 = 0.001
        self.learning_rate_2 = 0.0005
        self.EPOCH = 500

        self.d_ = np.zeros(self.EPOCH)
        self.s_ = np.zeros(self.EPOCH)
        self.W = np.zeros(self.EPOCH)

        self.up_weight_cliping_limit = 2.0
        self.dowm_weight_cliping_limit = 0.1
        self.weight_cliping_limit = 0.03

        self.s_optimizer = torch.optim.RMSprop(self.model.parameters(),lr=self.learning_rate_1)
        self.d_optimizer = torch.optim.RMSprop(self.D.parameters(),lr=self.learning_rate_2)
        self.criterion = nn.L1Loss()

        self.batchsize = 10
        self.lambda_term = 10


    def calculate_gradient_penalty(self,real_data,simu_data):
        eta = torch.FloatTensor(self.batchsize, 1, 1).uniform_(0, 1)
        eta = eta.expand(self.batchsize, real_data.size(1), real_data.size(2))
        if self.cuda:
            eta = eta.cuda()
        else:
            eta = eta

        interpolated = eta * real_data + ((-eta + 1) * simu_data)

        if self.cuda:
            interpolated = interpolated.cuda()
        else:
            interpolated = interpolated

        # define it to calculate gradient
        interpolated = Variable(interpolated, requires_grad=True)

        # calculate probability of interpolated examples
        prob_interpolated = self.D(interpolated)

        #calculate gradients of probabilities with respect to examples

        gradients1 = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                                  grad_outputs=torch.ones(
                                      prob_interpolated.size()).cuda() if self.cuda else torch.ones(
                                      prob_interpolated.size()),
                                  create_graph=True, retain_graph=True,only_inputs=True)[0]

        gradients2 = autograd.grad(outputs=gradients1, inputs=interpolated,
                                   grad_outputs=torch.ones(
                                       gradients1.size()).cuda() if self.cuda else torch.ones(
                                       gradients1.size()),
                                   create_graph=True, retain_graph=True)[0]

        grad_penalty = ((gradients2.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_term
        return grad_penalty

    def train(self):

        one = torch.tensor([1.])
        mone = one*-1
        if self.cuda:
            one = one.cuda()
            mone = mone.cuda()

        for iter in range(self.EPOCH):

            # model requires_grad=False
            for p in self.D.parameters():
                p.requires_grad = True


            for p in self.model.parameters():
                p.data.clamp_(self.dowm_weight_cliping_limit, self.up_weight_cliping_limit)

            d_cost = 0
            Wasserstein_D = 0


            for d_iter in range(5):
                # Train Discriminator
                self.D.zero_grad()
                for p in self.D.parameters():
                    p.data.clamp_(-self.weight_cliping_limit, self.weight_cliping_limit)

                real_data = self.model.Realnet((1+iter) * (1+d_iter))
                simu_data = self.model((1+iter)*(1+d_iter))

                if self.cuda:
                    real_data, simu_data = real_data.cuda(), simu_data.cuda()

                d_loss_real = self.D(real_data)
                d_loss_real = d_loss_real.mean()
                d_loss_real.backward(mone)

                d_loss_simu = self.D(simu_data)
                d_loss_simu = d_loss_simu.mean()
                d_loss_simu.backward(one)

                # gradient_penalty = self.calculate_gradient_penalty(real_data, simu_data)
                # gradient_penalty.backward()

                d_cost = d_loss_simu - d_loss_real # + gradient_penalty
                Wasserstein_D = d_loss_real - d_loss_simu
                self.d_optimizer.step()

                self.d_[iter] = d_cost
                self.W[iter] = Wasserstein_D


            #model update
            for p in self.D.parameters():
                p.requires_grad = False

            self.model.zero_grad()
            self.D.zero_grad()
            simu_data = self.model(iter)
            if self.cuda:
                simu_data = simu_data.cuda()

            s_loss = self.D(simu_data)
            s_loss = s_loss.mean()
            s_loss.backward(mone)
            s_cost = -s_loss
            self.s_optimizer.step()

            self.s_[iter] = s_cost

            if iter % 5 == 0:
                print('Epoch:{}, d_Loss:{}, s_Loss:{}, Wasserstein_D:{}'.format(iter,d_cost,s_cost,Wasserstein_D ))
                for name,param in self.model.named_parameters():
                    if param.requires_grad:
                        print(param)

            if iter == 300:
                plt.subplot(311)
                plt.plot(np.arange(300),self.d_[0:300])
                plt.subplot(312)
                plt.plot(np.arange(300), self.s_[0:300])
                plt.subplot(313)
                plt.plot(np.arange(300), self.W[0:300])
                plt.title('iter = 300,original2')
                plt.show()

            if iter == 500:
                plt.subplot(311)
                plt.plot(np.arange(500),self.d_[0:500])
                plt.subplot(312)
                plt.plot(np.arange(500), self.s_[0:500])
                plt.subplot(313)
                plt.plot(np.arange(500), self.W[0:500])
                plt.title('iter = 500,original2')
                plt.show()


def main():
    model = Simulator()
    model.train()

if __name__ == '__main__':
    main()
