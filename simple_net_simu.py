import ciw
import torch
import torch.nn as nn
import random


def recs_parser(recs):
    '''
    :param recs: object from the method:get_all_records
    :return: tensor which consists of data in param
    '''
    arri_t = [r.arrival_date for r in recs]
    wai_t = [r.waiting_time for r in recs]
    ser_start_t = [r.service_start_date for r in recs]
    ser_t = [r.service_time for r in recs]
    t_block = [r.time_blocked for r in recs]
    e_t = [r.exit_date for r in recs]
    result = arri_t + wai_t + ser_start_t + ser_t + t_block + e_t
    length = len(result)
    ciw_data = torch.zeros(length)
    for i in range(length):
        ciw_data[i] = result[i]
    return ciw_data


class CQnet(nn.Module):
    def __init__(self):
        super(CQnet,self).__init__()
        self.params0 = nn.Parameter(torch.Tensor([0.9]))
        self.params1 = nn.Parameter(torch.Tensor([0.9]))

    def forward(self,epoch):
        N = ciw.create_network(
            arrival_distributions=[ciw.dists.Exponential(self.params0)],
            service_distributions=[ciw.dists.Exponential(self.params1)],
            number_of_servers=[3]
        )
        ciw.seed(random.randint(0,epoch*1000))
        Q = ciw.Simulation(N)
        Q.simulate_until_max_customers(200)
        recs = Q.get_all_records()
        ciw_data = recs_parser(recs)
        return ciw_data



class Simu(object):
    def __init__(self):
        self.model = CQnet()
        self.learning_rate = 0.01
        self.EPOCH = 30
        self.batch_size = 10
        self.up_weight_cliping_limit = 1.0
        self.dowm_weight_cliping_limit = 0.001
        self.optimizer = torch.optim.RMSprop(self.model.parameters(),lr=self.learning_rate)
        self.criterion = nn.HingeEmbeddingLoss(margin=-1.0)
        self.N = ciw.create_network(
            arrival_distributions=[ciw.dists.Exponential(0.3)],
            service_distributions=[ciw.dists.Exponential(1.0)],
            number_of_servers=[3]
        )
        self.cus_num = 200

    def train(self):

        for iter in range(self.EPOCH):
            self.optimizer.zero_grad()
            for p in self.model.parameters():
                p.data.clamp_(self.dowm_weight_cliping_limit,self.up_weight_cliping_limit)

            simu_data = self.model(iter)
            ciw.seed(random.randint(1, 10000))
            Q = ciw.Simulation(self.N)
            Q.simulate_until_max_customers(self.cus_num)
            recs = Q.get_all_records()
            real_data = recs_parser(recs)
            for epoch in range(self.batch_size-1):
                ciw.seed(random.randint(1, 10000))
                Q = ciw.Simulation(self.N)
                Q.simulate_until_max_customers(self.cus_num)
                recs = Q.get_all_records()
                simu_data = simu_data + self.model(epoch)
                real_data = real_data + recs_parser(recs)
            simu_data = simu_data/self.batch_size
            real_data = real_data/self.batch_size
            distance = -torch.abs(simu_data - real_data)
            num = len(distance)
            label = torch.ones(num)*-1
            loss = self.criterion(distance,label)
            loss.backward()
            self.optimizer.step()
            if iter % 5 == 0:
                print('Epoch:{}, Loss:{}'.format(iter, loss.item()))
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        print(name, param)

def main():
    model = Simu()
    model.train()

if __name__ == '__main__':
    main()


