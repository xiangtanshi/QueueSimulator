import ciw
import torch
import torch.nn as nn
import random


class Simunet(nn.Module):
    def __init__(self):
        super(Simunet,self).__init__()
        self.param = torch.tensor([0.2,0.1,0.4,0.3,0.6])
        self.params1 = nn.Parameter(torch.Tensor([0.5,0.5]))
        self.params2 = nn.Parameter(torch.Tensor([0.5]))
        self.params3 = nn.Parameter(torch.Tensor([0.5,0.5,0]))

    def seq_process(self,seq,flag,recs):
        length = len(seq)

        if length <110:
            print('length:',length)
            raise KeyboardInterrupt
        elif flag == 1:
            for i in range(1,length):
                seq[i] = seq[i] - seq[0]
            return seq[1:101]
        else:
            seq.sort()
            return seq[1:101]

    def recs_parser(self,recs, node_num,cus_num):
        '''
        take collected object records from ciw's simulation and sort out the data in each node
        :param recs:
        :return: the list containing statistical info of each node's data
        '''
        arri_info = []
        wait_info = []
        ser_s_info = []
        ser_info = []
        ser_e_info = []
        block_info = []

        for i in range(1, node_num + 1):
            node_arri = {r.id_number:r.arrival_date for r in recs if r.node == i and r.id_number < cus_num-5}

            node = sorted(node_arri.items(),key=lambda item:item[1])
            index = [x[0] for x in node]

            arri = [x[1] for x in node]
            e = self.seq_process(arri,1,recs)
            arri_info.extend(e)

            node_wait = {r.id_number:r.waiting_time for r in recs if r.node == i and r.id_number < cus_num-5}
            wait = [node_wait[x] for x in index]
            e = self.seq_process(wait,2,recs)
            wait_info.extend(e)

            node_ser_s = {r.id_number:r.service_start_date for r in recs if r.node == i and r.id_number < cus_num-5}
            ser_s = [node_ser_s[x] for x in index]
            e = self.seq_process(ser_s,1,recs)
            ser_e_info.extend(e)

            node_ser_e = {r.id_number:r.service_end_date for r in recs if r.node == i and r.id_number < cus_num-5}
            ser_e = [node_ser_e[x] for x in index]
            e = self.seq_process(ser_e,1,recs)
            ser_s_info.extend(e)

            node_ser = {r.id_number:r.service_time for r in recs if r.node == i and r.id_number < cus_num-5}
            ser = [node_ser[x] for x in index]
            e = self.seq_process(ser,2,recs)
            ser_info.extend(e)

            node_block = {r.id_number:r.time_blocked for r in recs if r.node == i and r.id_number < cus_num-5}
            block = [node_block[x] for x in index]
            e = self.seq_process(block,2,recs)
            block_info.extend(e)

        ciw_data = arri_info + wait_info + ser_s_info + ser_e_info + ser_info + block_info
        num = len(ciw_data)
        result = torch.zeros(num)
        for i in range(num):
            result[i] = ciw_data[i]
        return result


    def forward(self,num_node):

        # N = ciw.create_network(
        #     arrival_distributions=[ciw.dists.Exponential(self.param[0])],
        #     service_distributions=[ciw.dists.Exponential(self.params[1])],
        #     number_of_servers=[3]
        # )

        N = ciw.create_network(
            arrival_distributions=[ciw.dists.Exponential(self.params1[0]),
                                   ciw.dists.Exponential(self.params1[1]),
                                   ciw.dists.NoArrivals()],
            service_distributions=[ciw.dists.Exponential(self.params2[0]),
                                   ciw.dists.Exponential(self.params3[0]),
                                   ciw.dists.Exponential(self.params3[1])],
            routing = [[0.0,0.3,0.7],
                       [0.0,0.0,1.0],
                       [0.0,0.0,0.0]],
            number_of_servers=[1,2,2]
        )
        ciw.seed(random.randint(0,100000))
        Q = ciw.Simulation(N)
        Q.simulate_until_max_customers(300)
        recs = Q.get_all_records()
        ciw_data = self.recs_parser(recs,num_node,300)
        return ciw_data

    def Realnet(self, num_node):

        # N = ciw.create_network(
        #     arrival_distributions=[ciw.dists.Exponential(self.param[0])],
        #     service_distributions=[ciw.dists.Exponential(self.param[1])],
        #     number_of_servers=[3]
        # )

        N = ciw.create_network(
            arrival_distributions=[ciw.dists.Exponential(self.param[0]),
                                   ciw.dists.Exponential(self.param[1]),
                                   ciw.dists.NoArrivals()],
            service_distributions=[ciw.dists.Exponential(self.param[2]),
                                   ciw.dists.Exponential(self.param[3]),
                                   ciw.dists.Exponential(self.param[4])],
            routing=[[0.0, 0.3, 0.7],
                     [0.0, 0.0, 1.0],
                     [0.0, 0.0, 0.0]],
            number_of_servers=[1, 2, 2]
        )
        ciw.seed(random.randint(0, 10000))
        Q = ciw.Simulation(N)
        Q.simulate_until_max_customers(300)
        recs = Q.get_all_records()
        ciw_data = self.recs_parser(recs, num_node,300)
        return ciw_data






class Simulator(object):
    def __init__(self):
        self.model = Simunet()
        self.learning_rate_1 = 0.004
        self.learning_rate_2 = 0.02
        self.learning_rate_3 = 0.001
        self.EPOCH = 200
        self.up_weight_cliping_limit = 1.0
        self.dowm_weight_cliping_limit = 0.001
        self.optimizer_1 = torch.optim.RMSprop(self.model.parameters(),lr=self.learning_rate_1)
        self.optimizer_2 = torch.optim.RMSprop(self.model.parameters(),lr=self.learning_rate_2)
        self.optimizer_3 = torch.optim.RMSprop(self.model.parameters(),lr=self.learning_rate_3)
        self.criterion = nn.SmoothL1Loss()
        self.node_num = 3
        self.cus_num = 300

    def train(self):

        for iter in range(self.EPOCH):
            self.optimizer_1.zero_grad()
            self.optimizer_2.zero_grad()
            for p in self.model.parameters():
                p.data.clamp_(self.dowm_weight_cliping_limit,self.up_weight_cliping_limit)
                if 115>iter>25 and len(p)==2:
                    p.requires_grad = False
                elif 115>iter>90 and len(p)==1:
                    p.requires_grad =False
                else:
                    p.requires_grad=True



            simu_data = self.model(self.node_num)
            real_data = self.model.Realnet(self.node_num)
            loss = self.criterion(simu_data,real_data)
            loss.backward()
            if iter<65:
                self.optimizer_1.step()
            elif 65<iter<130 and iter%5==0:
                self.optimizer_2.step()
            else:
                self.optimizer_3.step()
            if iter % 5 == 0:
                print('Epoch:{}, Loss:{}'.format(iter, loss.item()))
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        print(param)
                print('')

def main():
    model = Simulator()
    model.train()

if __name__ == '__main__':
    main()






