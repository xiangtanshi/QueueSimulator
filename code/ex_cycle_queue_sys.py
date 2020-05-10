import ciw
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class Simunet(nn.Module):
    def __init__(self):
        super(Simunet,self).__init__()
        self.params_a = nn.Parameter(torch.Tensor(1).uniform_(0.3,1.3))    #arrival params
        self.params_s = nn.Parameter(torch.Tensor(4).uniform_(0.3,1.3))    #service params


    def seq_process(self,seq):
        length = len(seq)
        if length<50:
            result = [0 for _ in range(50)]
            result[0:length] = seq
        else:
            result = seq[0:50]
        return result



    def recs_parser(self,recs,node_num,period):
        '''
         take collected object records from ciw's simulation and sort out the data in each node
        :param period denotes the epoch that is currently training, peroid start from 1
        :return: the list containing statistical info of each node's data
        '''

        parser_data = torch.zeros((1,node_num*2*50))  #feature dimension=1  (node_num * 2 * 50)
        lower_bound = 100 * (period-1)
        upper_bound = 100 * period
        for i in range(1, node_num + 1):

                node_arri = [r.arrival_date for r in recs if r.node == i and lower_bound<=r.arrival_date<=upper_bound]
                node_arri.sort()
                for j in range(len(node_arri)):
                    node_arri[j] = node_arri[j] - node_arri[0]
                arri = self.seq_process(node_arri)

                node_exit = [r.exit_date for r in recs if r.node == i and lower_bound<=r.arrival_date < upper_bound]
                node_exit.sort()
                for j in range(len(node_exit)):
                    node_exit[j] = node_exit[j] - node_exit[0]
                exit = self.seq_process(node_exit)

                feature = arri + exit

                for k in range(2*50):
                    parser_data[0][(i-1)*100 + k] = feature[k]

        return parser_data


    def forward(self,seed,period):

        N = ciw.create_network(
            arrival_distributions=[ciw.dists.Exponential(self.params_a[0]),
                                   ciw.dists.NoArrivals(),
                                   ciw.dists.NoArrivals(),
                                   ciw.dists.NoArrivals(),
                                   ciw.dists.NoArrivals()],
            service_distributions=[ciw.dists.Exponential(self.params_s[0]),
                                   ciw.dists.Lognormal(self.params_s[1],0.2),
                                   ciw.dists.Lognormal(self.params_s[2],0.2),
                                   ciw.dists.Exponential(self.params_s[3]),
                                   ciw.dists.Deterministic(0.001)],
            routing=[[0.0, 0.3, 0.7, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 0.5, 0.5],
                     [0.0, 0.0, 0.0, 0.2, 0.8],
                     [1.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 0.0]],
            number_of_servers=[2,1,1,2,4]
        )
        #simulate 10 times, 10 is the batchsize
        batch = torch.zeros(7,1,400)
        for i in range(7):
            ciw.seed(seed+i)
            Q = ciw.Simulation(N)
            Q.simulate_until_max_time(period*100+10)
            recs = Q.get_all_records()
            batch[i][0] = self.recs_parser(recs,4,period)
        return batch

    def Realnet(self,seed,period):

        N = ciw.create_network(
            arrival_distributions=[ciw.dists.Exponential(0.8),
                                   ciw.dists.NoArrivals(),
                                   ciw.dists.NoArrivals(),
                                   ciw.dists.NoArrivals(),
                                   ciw.dists.NoArrivals()],
            service_distributions=[ciw.dists.Exponential(0.4),
                                   ciw.dists.Lognormal(torch.tensor(0.9),torch.tensor(0.2)),
                                   ciw.dists.Lognormal(torch.tensor(0.4),torch.tensor(0.2)),
                                   ciw.dists.Exponential(2.0),
                                   ciw.dists.Deterministic(0.001)],
            routing=[[0.0, 0.3, 0.7, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 0.5, 0.5],
                     [0.0, 0.0, 0.0, 0.2, 0.8],
                     [1.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 0.0]],
            number_of_servers=[2, 1, 1, 2, 4]
        )
        batch = torch.zeros(7 ,1 , 400)
        for i in range(7):
            ciw.seed(seed + i)
            Q = ciw.Simulation(N)
            Q.simulate_until_max_time(period*100+10)
            recs = Q.get_all_records()
            batch[i][0] = self.recs_parser(recs, 4, period)
        return batch


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main_module = nn.Sequential(
            # state=7*1*400
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5, stride=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # state=7*1*133
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5, stride=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # state=7*1*44
            nn.Linear(in_features=44, out_features=1),
        )


    def forward(self,x):
        out = self.main_module(x)
        return out


class Simulator(object):
    def __init__(self):
        self.model = Simunet()
        self.D = Discriminator()
        if torch.cuda.is_available():
            self.cuda = True
            self.D = self.D.cuda()

        self.learning_rate_1 = 0.002
        self.learning_rate_2 = 0.00005
        self.EPOCH = 1000

        self.d_ = np.zeros(self.EPOCH)
        self.s_ = np.zeros(self.EPOCH)
        self.W = np.zeros(self.EPOCH)

        self.up_weight_cliping_limit = 2.0
        self.dowm_weight_cliping_limit = 0.15
        self.weight_cliping_limit = 0.03

        self.s_optimizer = torch.optim.RMSprop(self.model.parameters(),lr=self.learning_rate_1)
        self.d_optimizer = torch.optim.RMSprop(self.D.parameters(),lr=self.learning_rate_2)

        self.batchsize = 10

    def train(self):

        one = torch.tensor([1.])
        mone = one*-1
        if self.cuda:
            one = one.cuda()
            mone = mone.cuda()

        for period in range(1,2):     #1 periods

            #train
            for iter in range(self.EPOCH):
                # update discriminator
                for p in self.D.parameters():
                    p.requires_grad = True

                for p in self.model.parameters():
                    p.data.clamp_(self.dowm_weight_cliping_limit, self.up_weight_cliping_limit)

                d_cost = 0
                Wasserstein_D = 0


                for d_iter in range(4):
                    # Train Discriminator
                    self.D.zero_grad()
                    for p in self.D.parameters():
                        p.data.clamp_(-self.weight_cliping_limit, self.weight_cliping_limit)

                    real_data = self.model.Realnet((1+iter) * (1+d_iter),period)
                    simu_data = self.model((1+iter)*(1+d_iter),period)

                    if self.cuda:
                        real_data, simu_data = real_data.cuda(), simu_data.cuda()

                    d_loss_real = self.D(real_data)
                    d_loss_real = d_loss_real.mean()
                    d_loss_real.backward(mone)

                    d_loss_simu = self.D(simu_data)
                    d_loss_simu = d_loss_simu.mean()
                    d_loss_simu.backward(one)

                    d_cost = d_loss_simu - d_loss_real
                    Wasserstein_D = d_loss_real - d_loss_simu
                    self.d_optimizer.step()

                    self.d_[iter] = d_cost
                    self.W[iter] = Wasserstein_D


                #model update
                for p in self.D.parameters():
                    p.requires_grad = False

                self.model.zero_grad()
                self.D.zero_grad()
                simu_data = self.model(iter,period)
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

                if iter == 499:
                    plt.subplot(311)
                    plt.plot(np.arange(500),self.d_[0:500])
                    plt.subplot(312)
                    plt.plot(np.arange(500), self.s_[0:500])
                    plt.subplot(313)
                    plt.plot(np.arange(500), self.W[0:500])
                    plt.title('iter = 500,cycle_ex')
                    plt.savefig('cycle_ex_500.jpg')

                if iter == 999:
                    plt.subplot(311)
                    plt.plot(np.arange(1000),self.d_[0:1000])
                    plt.subplot(312)
                    plt.plot(np.arange(1000), self.s_[0:1000])
                    plt.subplot(313)
                    plt.plot(np.arange(1000), self.W[0:1000])
                    plt.title('iter = 1000,cycle_ex')
                    plt.savefig('cycle_ex_1000.jpg')


def main():
    model = Simulator()
    model.train()

if __name__ == '__main__':
    main()
