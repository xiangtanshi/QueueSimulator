#ultimate version
import ciw
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class Simunet(nn.Module):
    def __init__(self):
        super(Simunet,self).__init__()
        self.params_a = nn.Parameter(torch.Tensor(2).uniform_(0.3,1.3))    #arrival params
        self.params_s1 = nn.Parameter(torch.Tensor(2).uniform_(0.3,1.3))    #service params
        self.params_s2 = nn.Parameter(torch.Tensor(4).uniform_(0.3, 1.8))  # service params gamma, this place we make a assumption that alpha>1


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
                    parser_data[0][(i-1)*100+k] = feature[k]

        return parser_data


    def forward(self,seed,period):

        N = ciw.create_network(
            arrival_distributions=[ciw.dists.Exponential(self.params_a[0]),
                                   ciw.dists.Exponential(self.params_a[1]),
                                   ciw.dists.NoArrivals(),
                                   ciw.dists.NoArrivals()],
            service_distributions=[ciw.dists.Exponential(self.params_s1[0]),
                                   ciw.dists.Exponential(self.params_s1[1]),
                                   ciw.dists.Lognormal(self.params_s2[0],self.params_s2[1]),
                                   ciw.dists.Lognormal(self.params_s2[2],self.params_s2[3])],
            routing=[[0.0, 0.0, 0.4, 0.6],
                     [0.0, 0.0, 0.8, 0.2],
                     [0.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0]],
            number_of_servers=[2,2,3,3]
        )
        #simulate 10 times, 10 is the batchsize
        batch = torch.zeros(10,1,400)
        for i in range(10):
            ciw.seed(seed+i)
            Q = ciw.Simulation(N)
            Q.simulate_until_max_time(period*100+20)
            recs = Q.get_all_records()
            batch[i] = self.recs_parser(recs,4,period)
        return batch

    def Realnet_data(self,period):
        # generate training data, size = 300
        N = ciw.create_network(
            arrival_distributions=[ciw.dists.Exponential(1.0),
                                   ciw.dists.Exponential(0.5),
                                   ciw.dists.NoArrivals(),
                                   ciw.dists.NoArrivals()],
            service_distributions=[ciw.dists.Exponential(0.7),
                                   ciw.dists.Exponential(0.3),
                                   ciw.dists.Lognormal(torch.tensor(0.4),torch.tensor(0.3)),
                                   ciw.dists.Lognormal(torch.tensor(0.8),torch.tensor(0.1))],
            routing=[[0.0, 0.0, 0.4, 0.6],
                     [0.0, 0.0, 0.8, 0.2],
                     [0.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0]],
            number_of_servers=[2, 2, 3, 3]
        )
        training_data = torch.zeros(300 ,1 ,400)
        for i in range(300):
            ciw.seed(i**2)
            Q = ciw.Simulation(N)
            Q.simulate_until_max_time(period*100+20)
            recs = Q.get_all_records()
            training_data[i] = self.recs_parser(recs, 4, period)
        return training_data

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main_module = nn.Sequential(
            # state=10*1*400
            nn.Conv1d(in_channels=1, out_channels=2, kernel_size=5, stride=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # state=10*2*133
            nn.Conv1d(in_channels=2, out_channels=4, kernel_size=5, stride=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # state=10*4*44
        )
        self.L = nn.Linear(in_features=4*44,out_features=1)

    def forward(self,x):
        out = self.main_module(x)
        out = out.view(10,4*44)
        out = self.L(out)
        return out


class Simulator(object):
    def __init__(self):
        self.model = Simunet()
        self.D = Discriminator()
        if torch.cuda.is_available():
            self.cuda = True
            self.D = self.D.cuda()

        # learning rate for generator
        self.learning_rate_1 = 0.001
        self.learning_rate_2 = 0.008
        # learning rate for discriminator
        self.learning_rate_3 = 0.0005
        self.EPOCH = 5000

        self.cost_g = np.zeros(self.EPOCH)
        self.cost_d = np.zeros(self.EPOCH*4)
        # for comprison, to track how the W_distance(between two differrent simulation both with ground truth parameters) changes as the training goes
        self.compare = np.zeros(self.EPOCH)

        self.up_weight_cliping_limit = 2.5
        self.dowm_weight_cliping_limit = 0.15
        self.weight_cliping_limit = 0.05

        self.s_optimizer1 = torch.optim.RMSprop(self.model.parameters(),lr=self.learning_rate_1)
        self.s_optimizer2 = torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rate_2)
        self.d_optimizer = torch.optim.RMSprop(self.D.parameters(),lr=self.learning_rate_3)

        self.batchsize = 10

    def sample(self,training_data,batchsize):
        # size of training_data :300*1*400
        index = np.random.randint(0,10000,batchsize)
        index = [i%300 for i in index]
        batch = torch.zeros(batchsize,1,400)
        for i in range(batchsize):
            batch[i] = training_data[index[i]]
        return batch

    def train(self):

        one = torch.tensor([1.])
        mone = one*-1
        if self.cuda:
            one = one.cuda()
            mone = mone.cuda()

        training_data = self.model.Realnet_data(1)

        for period in range(1,2):

            #train
            for iter in range(self.EPOCH):
                # update discriminator
                for p in self.D.parameters():
                    p.requires_grad = True

                for p in self.model.parameters():
                    p.data.clamp_(self.dowm_weight_cliping_limit, self.up_weight_cliping_limit)

                Wasserstein_D = 0

                for d_iter in range(4):
                    # Train Discriminator
                    self.D.zero_grad()
                    for p in self.D.parameters():
                        p.data.clamp_(-self.weight_cliping_limit, self.weight_cliping_limit)

                    simu_data = self.model((1 + iter) * (1 + d_iter), period)
                    real_data = self.sample(training_data,self.batchsize)

                    if self.cuda:
                        real_data, simu_data = real_data.cuda(), simu_data.cuda()

                    d_loss_real = self.D(real_data)
                    d_loss_real = d_loss_real.mean()
                    d_loss_real.backward(mone)

                    d_loss_simu = self.D(simu_data)
                    d_loss_simu = d_loss_simu.mean()
                    d_loss_simu.backward(one)

                    Wasserstein_D = d_loss_real - d_loss_simu
                    # Wasserstein_Distance measures the difference between real queueing net and the simulating one in  parameters
                    # we persue larger W_distance by increasing d_loss_real and decreasing d_loss_simu
                    self.d_optimizer.step()

                    self.cost_d[4*iter+d_iter] = -Wasserstein_D


                #model update
                for p in self.D.parameters():
                    p.requires_grad = False

                self.model.zero_grad()
                simu_data = self.model(iter,period)
                if self.cuda:
                    simu_data = simu_data.cuda()

                s_loss = self.D(simu_data)
                s_loss = s_loss.mean()
                s_loss.backward(mone)
                # s_loss meatures the performance of the generator , we persue larger s_loss

                if iter%10 != 0:
                    self.s_optimizer1.step()
                else:
                    self.s_optimizer2.step()

                self.cost_g[iter] = -s_loss

                # test the discriminator
                true_data1 = self.sample(training_data, self.batchsize)
                true_data2 = self.sample(training_data, self.batchsize)
                if self.cuda:
                    true_data1,true_data2 = true_data1.cuda(), true_data2.cuda()
                d1 = self.D(true_data1).mean()
                d2 = self.D(true_data2).mean()
                self.compare[iter] = torch.abs(d1-d2)

                # disp the temperal result
                if iter % 10 == 0:
                    print('Epoch:{}, d_cost:{}, s_cost:{}'.format(iter,-Wasserstein_D,-s_loss))
                    for name,param in self.model.named_parameters():
                        if param.requires_grad:
                            print(name,param)
                    print('\n')

                if iter%1000 == 999:
                    plt.figure(int((iter+1)/1000))
                    ax1 = plt.subplot(311)
                    ax1.plot(np.arange(iter+1),self.cost_g[0:iter+1])
                    plt.title('g_cost')
                    ax2 = plt.subplot(312)
                    ax2.plot(np.arange((iter+1)*4), self.cost_d[0:(iter+1)*4])
                    plt.title('d_cost')
                    ax3 = plt.subplot(313)
                    ax3.plot(np.arange(iter+1),self.compare[0:iter+1])
                    plt.title('discriminating error')
                    plt.savefig('lognormal_iter_{}.jpg'.format(iter))

def main():
    model = Simulator()
    model.train()

if __name__ == '__main__':
    main()
