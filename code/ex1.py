import ciw
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class Simunet(nn.Module):
    def __init__(self):
        super(Simunet,self).__init__()
        #params of the distributions in the queueing network, uniformly initialized between (0.3,1.3)
        self.params = nn.Parameter(torch.Tensor(8).uniform_(0.0,1.5))

    def recs_parser(self,recs,node_num,length):
        '''
        recs is the detailed information of all customers that go through the network
        length is an approximation of the average number that each node serves
        :return: one dimension feature tensor that contains the time that each node starts service/end service with a customer
        '''
        parser_data = torch.zeros(node_num*length*2)  # feature dimension=1
        time_s = 0
        time_e = 100  #simulation lasts 200 time-unit

        for i in range(1, node_num + 1):

                node_arri = [r.arrival_date for r in recs if r.node == i and time_s<=r.arrival_date<=time_e]
                node_arri.sort()


                node_exit = [r.exit_date for r in recs if r.node == i and time_s<=r.arrival_date < time_e]
                node_exit.sort()

                index1 = (i-1)*length*2
                index2 = index1 + length
                for j in range(min(len(node_arri),len(node_exit),length)):
                    parser_data[index1+j] = node_arri[j] - node_arri[0]
                    parser_data[index2+j] = node_exit[j] - node_exit[0]

        return parser_data


    def forward(self,seed):

        N = ciw.create_network(
            arrival_distributions=[
                ciw.dists.Exponential(self.params[0]),
                ciw.dists.NoArrivals(),
                ciw.dists.NoArrivals(),
                ciw.dists.NoArrivals(),
                ciw.dists.NoArrivals()],
            service_distributions=[ciw.dists.Lognormal(self.params[1],self.params[2]),
                                   ciw.dists.Lognormal(self.params[3],self.params[4]),
                                   ciw.dists.Lognormal(self.params[5],self.params[6]),
                                   ciw.dists.Exponential(self.params[7]),
                                   ciw.dists.Deterministic(0.05)],
            routing=[[0.0, 0.8, 0.0, 0.2, 0.0],
                     [0.0, 0.0, 0.7, 0.0, 0.3],
                     [0.0, 0.0, 0.0, 0.5, 0.5],
                     [0.2, 0.8, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 0.0]],
            number_of_servers=[3, 3, 3, 2, 4]
        )
        #batchsize == 10
        batch = torch.zeros(10,1,1000)
        for i in range(10):
                ciw.seed(seed+i)
                Q = ciw.Simulation(N)
                Q.simulate_until_max_time(110)
                recs = Q.get_all_records()
                batch[i] = self.recs_parser(recs,5,100)
        return batch

    def Realnet_data(self):
        N = ciw.create_network(
            arrival_distributions=[
                                   ciw.dists.Exponential(1.0),
                                   ciw.dists.NoArrivals(),
                                   ciw.dists.NoArrivals(),
                                   ciw.dists.NoArrivals(),
                                   ciw.dists.NoArrivals()],
            service_distributions=[ciw.dists.Lognormal(torch.tensor(-0.2),torch.tensor(0.1)),
                                   ciw.dists.Lognormal(torch.tensor(1.0),torch.tensor(0.2)),
                                   ciw.dists.Lognormal(torch.tensor(0.0),torch.tensor(0.5)),
                                   ciw.dists.Exponential(1.0),
                                   ciw.dists.Deterministic(0.05)],
            routing=[[0.0, 0.8, 0.0, 0.2, 0.0],
                     [0.0, 0.0, 0.7, 0.0, 0.3],
                     [0.0, 0.0, 0.0, 0.5, 0.5],
                     [0.2, 0.8, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 0.0]],
            number_of_servers=[3, 3, 3, 2, 4]
        )

        num = 20
        training_data = torch.zeros(num, 1, 1000)
        for i in range(num):
            ciw.seed(i)
            Q = ciw.Simulation(N)
            Q.simulate_until_max_time(110)
            recs = Q.get_all_records()
            training_data[i] = self.recs_parser(recs, 5, 100)
        return  training_data

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main_module = nn.Sequential(
            #state = 10*1*1000
            nn.Conv1d(in_channels=1,out_channels=5,kernel_size=5,stride=3),
            nn.LeakyReLU(0.1,inplace=True),
            #state = 10*4*332
            nn.Conv1d(in_channels=5,out_channels=1,kernel_size=5,stride=3),
            nn.LeakyReLU(0.1,inplace=True),
            #state =10*1*110
        )
        self.L = nn.Linear(in_features=110,out_features=10)

    def forward(self,x):
        out = self.main_module(x)
        out = out.view(10,-1)
        out = self.L(out)
        return out

class Simulator(object):
    def __init__(self):
        self.model = Simunet()
        self.D = Discriminator()
        if torch.cuda.is_available():
            self.cuda = True
            self.D = self.D.cuda()
            #it is inefficient to put the simulation of ciw on cuda, as the computation of ciw is quite random and there are other operations in it
            # so we just put the discriminator on cuda

        # learning rate for generator
        self.learning_rate_1 = 0.003
        # learning rate for discriminator
        self.learning_rate_2 = 0.0001
        self.EPOCH = 1500

        self.d_real = np.zeros(self.EPOCH)
        self.d_simu = np.zeros(self.EPOCH)
        self.cost_d = np.zeros(self.EPOCH)
        # for comprison, to track how the Wasserstein distance changes when the inputs of disciminator are both training data as the training goes
        self.compare = np.zeros(self.EPOCH)

        self.weight_cliping_limit = 0.1

        self.s_optimizer1 = torch.optim.RMSprop(self.model.parameters(),lr=self.learning_rate_1)
        self.d_optimizer = torch.optim.RMSprop(self.D.parameters(),lr=self.learning_rate_2)
        self.s_scheduler = torch.optim.lr_scheduler.StepLR(self.s_optimizer1,step_size=200,gamma=0.5)
        self.d_scheduler = torch.optim.lr_scheduler.StepLR(self.d_optimizer,step_size=200,gamma=0.5)
        self.batchsize = 10
        self.lamda = 10

    def sample(self,data,batchsize):
        A,B,C = data.shape
        index = np.random.randint(0,1000,batchsize)
        index = [i%A for i in index]
        batch = torch.zeros(batchsize,1,C)
        for i in range(batchsize):
            batch[i] = data[index[i]]
        return batch

    def gradient_penalty(self,real_data,simu_data):
        '''calculate the gradient penalty item
        when the data pass through the discriminator network, the deeper the network, the more apparent gradients vanishes
        empically the layer of the discriminator network is 3(2conv+1fc), and the gradients is about O(e-6)
        one weird phenomenon is that once there are >=2 conv layers, we can not backpropagate the gradients of grad_penalty
        '''
        eta = torch.rand(self.batchsize,1,1)
        eta = eta.expand(real_data.size())
        if self.cuda:
            eta = eta.cuda()

        interpolated = eta * real_data + (torch.tensor(1.0) - eta) * simu_data
        if self.cuda:
            interpolated = interpolated.cuda()

        interpolated.requires_grad = True
        prob_interpolated = self.D(interpolated)
        gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                      grad_outputs=torch.ones(
                          prob_interpolated.size()).cuda() if self.cuda else torch.ones(
                          prob_interpolated.size()),
                      create_graph=True, retain_graph=True)[0]
        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lamda
        return grad_penalty

    def train(self):

        one = torch.tensor([1.])
        mone = one*-1
        if self.cuda:
            one = one.cuda()
            mone = mone.cuda()

        training_data = self.model.Realnet_data()

        for iter in range(self.EPOCH):
                # update discriminator
                for p in self.D.parameters():
                    p.requires_grad = True

                for p in self.model.parameters():
                    p.requires_grad = False


                for d_iter in range(2):
                    # Train the Discriminator
                    self.D.zero_grad()
                    for p in self.D.parameters():
                        p.data.clamp_(-self.weight_cliping_limit, self.weight_cliping_limit)

                    simu_data = self.model((1 + iter) * (1 + d_iter))
                    real_data = self.sample(training_data,self.batchsize)

                    if self.cuda:
                        real_data, simu_data = real_data.cuda(), simu_data.cuda()

                    d_loss_simu = self.D(simu_data).mean()
                    d_loss_simu.backward(one)

                    d_loss_real = self.D(real_data).mean()
                    d_loss_real.backward(mone)

                    Wasserstein_D = d_loss_real - d_loss_simu
                    # Wasserstein_Distance measures the difference of the distribution between the real_data and simu_data
                    # here in the point of discriminator, we persue larger W_distance by increasing d_loss_real and decreasing d_loss_simu
                    self.d_optimizer.step()

                    self.cost_d[iter] = Wasserstein_D + self.cost_d[iter]
                    self.d_real[iter] = d_loss_real + self.d_real[iter]
                    self.d_simu[iter] = d_loss_simu + self.d_simu[iter]

                self.cost_d[iter] /= 2
                self.d_real[iter] /= 2
                self.d_simu[iter] /= 2

                #update the distribution params of the model
                for p in self.D.parameters():
                    p.requires_grad = False
                for p in self.model.parameters():
                    p.requires_grad = True

                self.model.zero_grad()
                simu_data = self.model(iter)
                if self.cuda:
                    simu_data = simu_data.cuda()

                s_loss = self.D(simu_data).mean()
                s_loss.backward(mone)
                # s_loss meatures the performance of the generator , we persue larger s_loss


                self.s_optimizer1.step()

                self.s_scheduler.step()
                self.d_scheduler.step()

                # test the discriminator
                true_data1 = self.sample(training_data, self.batchsize)
                true_data2 = self.sample(training_data, self.batchsize)
                if self.cuda:
                    true_data1,true_data2 = true_data1.cuda(), true_data2.cuda()
                d1 = self.D(true_data1).mean()
                d2 = self.D(true_data2).mean()
                self.compare[iter] = d1-d2

                # disp the temperal result
                if iter % 5 == 0:
                    print('Epoch:{}, d_real:{}, d_simu:{}, wasserstein:{}, compare:{}'.format(iter,self.d_real[iter],self.d_simu[iter],self.cost_d[iter],self.compare[iter]))
                    for name,param in self.model.named_parameters():
                        if param.requires_grad:
                            print(name,param)
                    print('\n')

                if iter == 599:
                    plt.figure(1)
                    plt.plot(np.arange((iter+1)), self.cost_d[0:iter+1],color='green',label='w_distance between real and simu data')
                    plt.plot(np.arange(iter+1),self.compare[0:iter+1],color='red',label='w_distance between two real data')
                    plt.plot(np.arange(iter+1),self.d_real[0:iter+1],color='blue',label='d_loss_real')
                    plt.plot(np.arange(iter+1),self.d_simu[0:iter+1],color='black',label='d_loss_simu')
                    plt.xlabel('time steps')
                    plt.ylabel('output value of Discriminator')
                    plt.legend()
                    plt.savefig('HPP_result_itr_600.jpg')
                if iter == 1499:
                    plt.figure(2)
                    plt.plot(np.arange((iter + 1)), self.cost_d[0:iter + 1], color='green',
                             label='w_distance between real and simu data')
                    plt.plot(np.arange(iter + 1), self.compare[0:iter + 1], color='red',
                             label='w_distance between two real data')
                    plt.plot(np.arange(iter + 1), self.d_real[0:iter+1], color='blue', label='d_loss_real')
                    plt.plot(np.arange(iter + 1), self.d_simu[0:iter+1], color='black', label='d_loss_simu')
                    plt.xlabel('time steps')
                    plt.ylabel('output value of Discriminator')
                    plt.legend()
                    plt.savefig('HPP_result_itr_1500.jpg')

def main():
    model = Simulator()
    model.train()

if __name__ == '__main__':
    main()