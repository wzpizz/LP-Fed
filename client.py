import time
import torch
from utils import get_optimizer, get_model
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.autograd import Variable
import copy
from optimization import Optimization
from server import Server
from uplayer import Sim_map
def add_model(dst_model, src_model, dst_no_data, src_no_data):
    if dst_model is None:
        result = copy.deepcopy(src_model)
        return result
    params1 = src_model.named_parameters()
    params2 = dst_model.named_parameters()
    dict_params2 = dict(params2)
    with torch.no_grad():
        for name1, param1 in params1:
            if name1 in dict_params2:
                dict_params2[name1].set_(param1.data*src_no_data + dict_params2[name1].data*dst_no_data)
    return dst_model


def scale_model(model, scale):
    params = model.named_parameters()
    dict_params = dict(params)
    with torch.no_grad():
        for name, param in dict_params.items():
            dict_params[name].set_(dict_params[name].data * scale)
    return model

def aggregate_models(models, weights):
    """aggregate models based on weights
    params:
        models: model updates from clients
        weights: weights for each model, e.g. by data sizes or cosine distance of features
    """
    if models == []:
        return None
    model = add_model(None, models[0], 0, weights[0])
    total_no_data = weights[0]
    for i in range(1, len(models)):
        model = add_model(model, models[i], total_no_data, weights[i])
        model = scale_model(model, 1.0 / (total_no_data+weights[i]))
        total_no_data = total_no_data + weights[i]
    return model
class Client():
    def __init__(self,  cid, data, device, project_dir, model_name, local_epoch, lr, batch_size, drop_rate, stride):

        self.cid = cid
        self.project_dir = project_dir
        self.model_name = model_name
        self.data = data
        self.device = device
        self.local_epoch = local_epoch
        self.lr = lr

        self.batch_size = batch_size
        # self.dataset_sizes = len(self.data.train_dataset_sizes[cid])
        self.dataset_sizes = self.data.train_dataset_sizes[cid]
        # self.dataset_sizes = self.data.preprocess_train()
        self.train_loader = self.data.train_loaders[cid]

        self.full_model = get_model(self.data.train_class_sizes[cid], drop_rate, stride)
        self.classifier = self.full_model.classifier.classifier
        self.full_model.classifier.classifier = nn.Sequential()
        self.model = self.full_model
        self.distance=0
        self.optimization = Optimization(self.train_loader, self.device)
        self.sim_map = Sim_map(self.train_loader, self.device)
        # print("class name size",class_names_size[cid])
        # train_data = self.data.train_loaders[cid]
        # self.eta = et
        # self.rand_percent = rp
        # self.layer_idx = li
        # self.loss = nn.CrossEntropyLoss()
        # self.ALA = ALA(self.cid, self.loss, train_data, self.batch_size,
        #                self.rand_percent, self.layer_idx, self.eta, self.device)

    def train(self, federated_model, use_cuda):
        self.y_err = []
        self.y_loss = []
        # # model.load_state_dict 加载模型



        self.model.load_state_dict(federated_model.state_dict())
        # print(federated_model)
        self.model.classifier.classifier = self.classifier
        self.old_classifier = copy.deepcopy(self.classifier)
        self.model = self.model.to(self.device)

        optimizer = get_optimizer(self.model, self.lr)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

        criterion = nn.CrossEntropyLoss()

        since = time.time()

        print('Client', self.cid, 'start training')
        for epoch in range(self.local_epoch):
            print('Epoch {}/{}'.format(epoch, self.local_epoch - 1))
            print('-' * 10)

            scheduler.step()
            self.model.train(True)
            running_loss = 0.0
            running_corrects = 0.0

            for data in self.train_loader:
                inputs, labels = data
                b, c, h, w = inputs.shape
                if b < self.batch_size:
                    continue
                if use_cuda:
                    inputs = Variable(inputs.cuda().detach())
                    labels = Variable(labels.cuda().detach())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()

                outputs = self.model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                loss.backward()

                optimizer.step()

                running_loss += loss.item() * b
                running_corrects += float(torch.sum(preds == labels.data))

            used_data_sizes = (self.dataset_sizes - self.dataset_sizes % self.batch_size)
            epoch_loss = running_loss / used_data_sizes
            epoch_acc = running_corrects / used_data_sizes

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                'train', epoch_loss, epoch_acc))

            self.y_loss.append(epoch_loss)
            self.y_err.append(1.0-epoch_acc)

            time_elapsed = time.time() - since
            print('Client', self.cid, ' Training complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))

        time_elapsed = time.time() - since
        print('Client', self.cid, 'Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))


        # save_network(self.model, self.cid, 'last', self.project_dir, self.model_name, gpu_ids)

        self.classifier = self.model.classifier.classifier
        self.distance = self.optimization.cdw_feature_distance(federated_model, self.old_classifier, self.model)
        self.model.classifier.classifier = nn.Sequential()


    # def train(self, federated_model, use_cuda):
    #     self.y_err = []
    #     self.y_loss = []
    #     self.model.classifier.classifier = self.classifier
    #     self.old_classifier = copy.deepcopy(self.classifier)
    #     # c=self.optimization.get_gexinghua(federated_model, self.old_classifier, self.model)
    #     # c=self.optimization.get_similarity_cca(federated_model, self.old_classifier, self.model)
    #     # c = self.optimization.mse_feature_distance(federated_model, self.old_classifier, self.model)
    #     c = self.sim_map.compute_similarity(federated_model, self.old_classifier, self.model)
    #     model1 = add_model(federated_model, self.model, 1-c, c)
    #     # print(c)
    #     # if(c > 0):
    #     #     # cL+(1-c)F
    #     # model1 = add_model(federated_model, self.model, 1 - c, c)
    #     # #     # (1-c)l+cf
    #     # model1 = add_model(federated_model, self.model, c ,1 - c)
    #     self.model.load_state_dict(model1.state_dict(), strict=False)
    #     # else:
    #     #     self.model.load_state_dict(federated_model.state_dict(), strict=False)
    #
    #
    #     self.model.classifier.classifier = self.classifier
    #     # self.old_classifier = copy.deepcopy(self.classifier)
    #     self.model = self.model.to(self.device)
    #
    #     optimizer = get_optimizer(self.model, self.lr)
    #     scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
    #
    #     criterion = nn.CrossEntropyLoss()
    #
    #     since = time.time()
    #
    #     print('Client', self.cid, 'start training')
    #     for epoch in range(self.local_epoch):
    #         print('Epoch {}/{}'.format(epoch, self.local_epoch - 1))
    #         print('-' * 10)
    #
    #         scheduler.step()
    #         self.model.train(True)
    #         running_loss = 0.0
    #         running_corrects = 0.0
    #
    #         for data in self.train_loader:
    #             inputs, labels = data
    #             b, c, h, w = inputs.shape
    #             if b < self.batch_size:
    #                 continue
    #             if use_cuda:
    #                 inputs = Variable(inputs.cuda().detach())
    #                 labels = Variable(labels.cuda().detach())
    #             else:
    #                 inputs, labels = Variable(inputs), Variable(labels)
    #
    #             optimizer.zero_grad()
    #
    #             outputs = self.model(inputs)
    #             _, preds = torch.max(outputs.data, 1)
    #             loss = criterion(outputs, labels)
    #             loss.backward()
    #
    #             optimizer.step()
    #
    #             running_loss += loss.item() * b
    #             running_corrects += float(torch.sum(preds == labels.data))
    #
    #         used_data_sizes = (self.dataset_sizes - self.dataset_sizes % self.batch_size)
    #         epoch_loss = running_loss / used_data_sizes
    #         epoch_acc = running_corrects / used_data_sizes
    #
    #         print('{} Loss: {:.4f} Acc: {:.4f}'.format(
    #             'train', epoch_loss, epoch_acc))
    #
    #         self.y_loss.append(epoch_loss)
    #         self.y_err.append(1.0-epoch_acc)
    #
    #         time_elapsed = time.time() - since
    #         print('Client', self.cid, ' Training complete in {:.0f}m {:.0f}s'.format(
    #             time_elapsed // 60, time_elapsed % 60))
    #
    #     time_elapsed = time.time() - since
    #     print('Client', self.cid, 'Training complete in {:.0f}m {:.0f}s'.format(
    #         time_elapsed // 60, time_elapsed % 60))
    #
    #
    #     # save_network(self.model, self.cid, 'last', self.project_dir, self.model_name, gpu_ids)
    #
    #     self.classifier = self.model.classifier.classifier
    #     self.distance = self.optimization.cdw_feature_distance(federated_model, self.old_classifier, self.model)
    #     self.model.classifier.classifier = nn.Sequential()



    # def train(self, federated_model,cdw, use_cuda):
    #     self.y_err = []
    #     self.y_loss = []
    #
    #
    #     # model.load_state_dict 加载模型
    #     #print(federated_model)
    #     self.model.classifier.classifier = self.classifier
    #     self.old_classifier = copy.deepcopy(self.classifier)
    #     # c=self.optimization.get_gexinghua(federated_model, self.old_classifier, self.model)
    #     # c=self.optimization.get_similarity_cca(federated_model, self.old_classifier, self.model)
    #     c = self.optimization.get_similarity_cka(federated_model, self.old_classifier, self.model)
    #     # print(c)
    #     # if(c > 0):
    #     #     # cL+(1-c)F
    #     model1 = add_model(federated_model, self.model, 1-c, c)
    #     # #     # (1-c)l+cf
    #     # model1 = add_model(federated_model, self.model, c ,1 - c)
    #     self.model.load_state_dict(model1.state_dict(), strict=False)
    #     # else:
    #     #     self.model.load_state_dict(federated_model.state_dict(), strict=False)
    #
    #     self.model.classifier.classifier = self.classifier
    #     # self.old_classifier = copy.deepcopy(self.classifier)
    #     self.model = self.model.to(self.device)
    #
    #     optimizer = get_optimizer(self.model, self.lr)
    #     scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
    #
    #     criterion = nn.CrossEntropyLoss()
    #
    #     since = time.time()
    #
    #     print('Client', self.cid, 'start training')
    #     for epoch in range(self.local_epoch):
    #         print('Epoch {}/{}'.format(epoch, self.local_epoch - 1))
    #         print('-' * 10)
    #
    #         scheduler.step()
    #         self.model.train(True)
    #         running_loss = 0.0
    #         running_corrects = 0.0
    #
    #         for data in self.train_loader:
    #             inputs, labels = data
    #             b, c, h, w = inputs.shape
    #             if b < self.batch_size:
    #                 continue
    #             if use_cuda:
    #                 inputs = Variable(inputs.cuda().detach())
    #                 labels = Variable(labels.cuda().detach())
    #             else:
    #                 inputs, labels = Variable(inputs), Variable(labels)
    #
    #             optimizer.zero_grad()
    #
    #             outputs = self.model(inputs)
    #             _, preds = torch.max(outputs.data, 1)
    #             loss = criterion(outputs, labels)
    #             loss.backward()
    #
    #             optimizer.step()
    #
    #             running_loss += loss.item() * b
    #             running_corrects += float(torch.sum(preds == labels.data))
    #
    #         used_data_sizes = (self.dataset_sizes - self.dataset_sizes % self.batch_size)
    #         epoch_loss = running_loss / used_data_sizes
    #         epoch_acc = running_corrects / used_data_sizes
    #
    #         print('{} Loss: {:.4f} Acc: {:.4f}'.format(
    #             'train', epoch_loss, epoch_acc))
    #
    #         self.y_loss.append(epoch_loss)
    #         self.y_err.append(1.0 - epoch_acc)
    #
    #         time_elapsed = time.time() - since
    #         print('Client', self.cid, ' Training complete in {:.0f}m {:.0f}s'.format(
    #             time_elapsed // 60, time_elapsed % 60))
    #
    #     time_elapsed = time.time() - since
    #     print('Client', self.cid, 'Training complete in {:.0f}m {:.0f}s'.format(
    #         time_elapsed // 60, time_elapsed % 60))
    #
    #     # save_network(self.model, self.cid, 'last', self.project_dir, self.model_name, gpu_ids)
    #
    #     self.classifier = self.model.classifier.classifier
    #     self.distance = self.optimization.cdw_feature_distance(federated_model, self.old_classifier, self.model)
    #     self.model.classifier.classifier = nn.Sequential()
    def generate_soft_label(self, x, regularization):
        return self.optimization.kd_generate_soft_label(self.model, x, regularization)

    def get_model(self):
        return self.model

    def get_data_sizes(self):
        return self.dataset_sizes

    def get_train_loss(self):
        return self.y_loss[-1]

    def get_cos_distance_weight(self):
        return self.distance