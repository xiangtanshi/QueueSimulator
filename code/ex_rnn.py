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

    def recs_parser(self,recs,node_num):
        '''
        recs is the detailed information of all customers that go through the network
        :return: a list which contains input sequence tensors of the discriminator's rnn
        '''
        parser_data = []
        time_s = 0.0
        time_e = 80.0  #simulation lasts 100 time-unit

        for i in range(1, node_num + 1):

                node_arri = [r.arrival_date/time_e for r in recs if r.node == i and time_s<=r.arrival_date<=time_e]
                node_arri.sort()
                seq_len1 = int(len(node_arri)//4)
                if seq_len1 == 0:
                    raise ValueError('customers should be more than 4')
                arrival_seq = torch.zeros(seq_len1,1,4)
                for row in range(seq_len1):
                    for col in range(4):
                        arrival_seq[row][0][col] = node_arri[row*4+col] - node_arri[row*4+col-1]
                arrival_seq[0][0][0] = node_arri[0]

                node_exit = [r.exit_date/time_e for r in recs if r.node == i and time_s<=r.arrival_date < time_e]
                node_exit.sort()
                seq_len2 = int(len(node_exit)//4)
                if seq_len2 == 0:
                    raise ValueError('customers should be more than 4')
                exit_seq = torch.zeros(seq_len2,1,4)
                for row in range(seq_len2):
                    for col in range(4):
                        exit_seq[row][0][col] = node_exit[row*4+col] - node_exit[row*4+col-1]
                exit_seq[0][0][0] = node_exit[0]

                parser_data.append(arrival_seq)
                parser_data.append(exit_seq)

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
            routing=[[0.0, 0.5, 0.0, 0.5, 0.0],
                     [0.0, 0.0, 0.7, 0.0, 0.3],
                     [0.0, 0.0, 0.0, 0.5, 0.5],
                     [0.2, 0.8, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 0.0]],
            number_of_servers=[5, 3, 2, 2, 4],
        )
        #batchsize == 20
        batch = []
        for i in range(10):
                ciw.seed(seed+i)
                Q = ciw.Simulation(N)
                Q.simulate_until_max_time(80)
                recs = Q.get_all_records()
                batch.append(self.recs_parser(recs,5))
        return batch

    def Realnet_data(self):
        N = ciw.create_network(
            arrival_distributions=[
                                   ciw.dists.Exponential(2.0),
                                   ciw.dists.NoArrivals(),
                                   ciw.dists.NoArrivals(),
                                   ciw.dists.NoArrivals(),
                                   ciw.dists.NoArrivals()],
            service_distributions=[ciw.dists.Lognormal(torch.tensor(-0.3),torch.tensor(0.1)),
                                   ciw.dists.Lognormal(torch.tensor(0.0),torch.tensor(0.5)),
                                   ciw.dists.Lognormal(torch.tensor(0.5),torch.tensor(0.1)),
                                   ciw.dists.Exponential(1.0),
                                   ciw.dists.Deterministic(0.05)],
            routing=[[0.0, 0.5, 0.0, 0.5, 0.0],
                     [0.0, 0.0, 0.7, 0.0, 0.3],
                     [0.0, 0.0, 0.0, 0.5, 0.5],
                     [0.2, 0.8, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 0.0]],
            number_of_servers=[5, 3, 2, 2, 4],
        )
        num = 200
        training_data = []
        for i in range(num):
            ciw.seed(i)
            Q = ciw.Simulation(N)
            Q.simulate_until_max_time(80)
            recs = Q.get_all_records()
            training_data.append(self.recs_parser(recs, 5))
        return  training_data

class Discriminator(nn.Module):
    def __init__(self,node_num):
        #assign a different rnn for each node sequence
        super().__init__()
        self.RNN = []
        self.num = node_num
        for _ in range(self.num):
            self.RNN.append(nn.RNN(input_size=4, hidden_size=4, num_layers=1, nonlinearity='relu'))
        #for attention
        self.L = nn.Linear(in_features=4,out_features=1)

    def forward(self,x):
        # x is the batch list
        batchsize = len(x)
        result = 0
        for k in range(batchsize):
            for i in range(self.num):
                #x[k][i]: seq_len, batchsize=1, dim=4
                output1,hn1 = self.RNN[i](x[k][2*i])
                output2,hn2 = self.RNN[i](x[k][2*i+1])
                #attention
                result += self.attention(output1,hn1)
                result += self.attention(output2,hn2)
        return result

    def attention(self,output,hn):
        #  apply attention mechanism to the rnn output, simplify the processing
        # seq_len = output.shape[0]
        # H = torch.zeros(seq_len,1,4)
        # for i in range(seq_len):
        #     H[i] = hn
        # data = torch.cat((output,H),dim=2)
        # result = self.L(data)
        # return torch.sum(result)
        result = self.L(output)
        return torch.sum(result)

class Simulator(object):
    def __init__(self):
        self.model = Simunet()
        self.D = Discriminator(5)
        # if torch.cuda.is_available():
        #     self.cuda = True
        #     self.D = self.D.cuda()
            #it is inefficient to put the simulation of ciw on cuda, as the computation of ciw is quite random and there are other operations in it
            # so we just put the discriminator on cuda
        self.cuda = False

        # learning rate for generator
        self.learning_rate_1 = 0.005
        # learning rate for discriminator
        self.learning_rate_2 = 0.0001
        self.EPOCH = 1000

        self.d_real = np.zeros(self.EPOCH)
        self.d_simu = np.zeros(self.EPOCH)
        self.cost_d = np.zeros(self.EPOCH)
        # for comprison, to track how the Wasserstein distance changes when the inputs of disciminator are both training data as the training goes
        self.compare = np.zeros(self.EPOCH)

        self.weight_cliping_limit = 0.05

        self.s_optimizer1 = torch.optim.RMSprop(self.model.parameters(),lr=self.learning_rate_1)
        self.d_optimizer = torch.optim.RMSprop(self.D.parameters(),lr=self.learning_rate_2,centered=True)
        self.s_scheduler = torch.optim.lr_scheduler.StepLR(self.s_optimizer1,step_size=200,gamma=0.5)
        self.d_scheduler = torch.optim.lr_scheduler.StepLR(self.d_optimizer,step_size=200,gamma=0.5)

        self.batchsize = 10

    def sample(self,data,batchsize):
        size = len(data)
        index = np.random.randint(0,200,batchsize)
        index = [i%size for i in index]
        batch = []
        for i in range(batchsize):
            batch.append(data[index[i]])
        return batch

    def Gradient_penalty(self,real_data,simu_data):
        pass

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

                # test the discriminator
                true_data1 = self.sample(training_data, self.batchsize)
                true_data2 = self.sample(training_data, self.batchsize)
                if self.cuda:
                    true_data1,true_data2 = true_data1.cuda(), true_data2.cuda()
                d1 = self.D(true_data1).mean()
                d2 = self.D(true_data2).mean()
                self.compare[iter] = d1-d2

                self.d_scheduler.step()
                self.s_scheduler.step()

                # disp the temperal result
                if iter % 5 == 0:
                    print('Epoch:{}, d_real:{}, d_simu:{}, wasserstein:{}, compare:{}'.format(iter,self.d_real[iter],self.d_simu[iter],self.cost_d[iter],self.compare[iter]))
                    for name,param in self.model.named_parameters():
                        if param.requires_grad:
                            print(name,param)
                    print('\n')

                if iter == 499:
                    plt.figure(1)
                    plt.plot(np.arange((iter+1)), self.cost_d[0:iter+1],color='green',label='w_distance between real and simu data')
                    plt.plot(np.arange(iter+1),self.compare[0:iter+1],color='red',label='w_distance between two real data')
                    plt.plot(np.arange(iter+1),self.d_real[0:iter+1],color='blue',label='d_loss_real')
                    plt.plot(np.arange(iter+1),self.d_simu[0:iter+1],color='black',label='d_loss_simu')
                    plt.xlabel('time steps')
                    plt.ylabel('output value of Discriminator')
                    plt.legend()
                    plt.savefig('HPP_itr_500.jpg')
                if iter == 999:
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
                    plt.savefig('HPP_itr_1000.jpg')

def main():
    model = Simulator()
    model.train()

if __name__ == '__main__':
    main()

