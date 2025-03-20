import torch
import torch.nn as nn
import torch.nn.functional as F

import models
import utils
from .models import register


def computeProto(x_shot, x_query):
    # 1.打印信息，确保正确拿到数据
    #print('coputeProto: ')
    #print('x_shot={}, {}, {}'.format(x_shot.shape, type(x_shot), x_shot.device.type))
    #print('x_query={}, {}, {}'.format(x_query.shape, type(x_query), x_query.device.type))

    # 2.定义接受所有batch原型的tensor
    cnt = x_shot.shape[1]
    proto_for_batch = torch.rand(1, cnt, 512).to('cuda:0')

    # 3.一种朴素的实现方法是：遍历每一个任务的每一个支持集类别，完成精细原型的计算，然后将它们拼接起来，返回进行下一步计算。
    # 现在，我们来遍历这些数据。
    for i in range(x_shot.shape[0]):    # 遍历每一个episode
        # 拿到该episode数据
        x_shot_of_episode = x_shot[i, :, :, :]
        #print('episode的shape=', x_shot_of_episode.shape)
        # print(x_shot_of_episode)

        # 准备一个空的tensor接精细原型，形状为[1,5]
        proto_for_episode = torch.rand(1, 512).to('cuda:0')
        for j in range(x_shot.shape[1]):    # 遍历每一个支持集类别
            # 1.处理支持集特征向量
            # 1.1 拿到该类别的所有特征向量
            x_shot_of_class = x_shot_of_episode[j, :, :]
            #print('1.1 该类别的原始数据shape=', x_shot_of_class.shape)
            # print(x_shot_of_class)

            # 1.2 对于每一个支持集类别，我们首先将该类所有样本特征向量相加，
            # x_shot_of_class = x_shot_of_class.sum(dim=0)
            # 将NumPy数组转换为PyTorch的Tensor
            # x_shot_of_class = torch.from_numpy(x_shot_of_class)

            #print('1.2 shape=', x_shot_of_class.shape)
            x_shot_of_class = torch.sum(x_shot_of_class, dim=0)
            #print('1.2 shape=', x_shot_of_class.shape)
            #print('1.2 该类别的数据总向量shape=', x_shot_of_class.shape)
            # print(x_shot_of_class)

            # 2.处理查询集特征向量
            # 2.1 计算类均值
            # x_shot_of_episode = torch.from_numpy(x_shot_of_episode)
            mean = x_shot_of_episode[j, :, :].mean(dim=-2)
            #print('2.1 类均值shape=', mean.shape)
            # print(mean)

            # 扩展tensor1的维度以便广播
            mean = mean.unsqueeze(0)  # 现在形状为[1, 5]
            #print('2.2 类均值扩展维度shape=', mean.shape)
            # print(mean)

            # 2.2 计算每一个查询样本与类均值的欧式距离二范数，然后得到的结构做softmax，得到的分数和查询样本做加权求和
            # 拿到所有查询样本特征向量
            x_query_of_episode = x_query[i, :, :]
            #print('2.2 查询样本shape=', x_query_of_episode.shape)
            # print(x_query_of_episode)

            # 二者计算欧式距离
            # 计算差的平方
            # x_query_of_episode = torch.from_numpy(x_query_of_episode)
            # mean = torch.from_numpy(mean)
            # print('mean type=', type(mean), type(x_query_of_episode))
            squared_diff = (mean - x_query_of_episode) ** 2

            # 求和得到平方差和
            sum_squared_diff = torch.sum(squared_diff, dim=1)  # 沿着第二个维度（原tensor1的扩展维度）求和

            # 开平方根得到欧式距离
            euclidean_distances = torch.sqrt(sum_squared_diff)

            #print('得到查询样本到类原型的距离:',euclidean_distances.shape)  # 应输出torch.Size([15]), 每个值对应tensor1到tensor2中一个向量的欧式距离
            # print(euclidean_distances)

            # 直接应用Softmax。注意，这里没有取负，因为通常Softmax用于正向比较，但根据实际应用场景，你可能需要调整。
            weights = F.softmax(-euclidean_distances, dim=0)  # 注意这里取了负号，转换为类似相似度的度量

            # 确认权重和为1
            # assert torch.isclose(weights.sum(), torch.tensor(1.0)), "权重之和不为1"

            # weights = weights.unsqueeze(0)
            #print('距离经softmax得到权重：', weights.shape)
            #print('查询样本：', x_query_of_episode.shape)
            # result = weights * x_query_of_episode

            result = weights[:, None] * x_query_of_episode
            #print('相乘得到结果：', result.shape)

            # 将加权后的向量相加
            result = result.sum(dim=0)
            #print('再相加最终结果：', result.shape)

            # 3.将步骤1和2得到的结果做一个相加
            # 分子 = 支持集特征向量和 + 查询集加权向量和
            #if x_shot.shape[2] == 1:
                #up = x_shot_of_class + 0.2 * result
            #else:
                #up = x_shot_of_class + result
            up = x_shot_of_class + 37*result

            #print('分子=', up.shape)

            # 4.计算分母，分母=查询集样本数量+1
            if x_shot.shape[2] == 1:
                b = 0.2
            else:
                b = 1
            b = 37
            down = x_shot.shape[2] + b
            #print('down=', down)

            # 5.分子除以分母，得到一个类别的精细原型，形状为[5]
            end = up / down
            #print('end', end.shape)

            end = end.unsqueeze(0)
            #print('end维度扩展, 准备叠加', end.shape)     # [1,5]

            # 多个类别的原型向量应该拼接起来。
            proto_for_episode = torch.cat((proto_for_episode, end), dim=0)
            pass

        # 去除第一个结果
        proto_for_episode = proto_for_episode[1:, :]
        #print('一集的叠加好的原型形状', proto_for_episode.shape)

        # 维度扩展, 准备多集加在一起做成一个batch的精细原型
        proto_for_episode = proto_for_episode.unsqueeze(0)
        # all = torch.cat((tensor1, tensor2), dim=0)
        # 多个episode的精细原型一共拼接起来
        #print('---------------------------')
        #print(proto_for_episode.shape, proto_for_batch.shape)
        #print('---------------------------')
        proto_for_batch = torch.cat((proto_for_batch, proto_for_episode), dim=0)
        pass
    # 去除第一个结果
    proto_for_batch = proto_for_batch[1:, :]
    #print('最终形状:', proto_for_batch.shape)
    return proto_for_batch
    pass


@register('meta-baseline')
class MetaBaseline(nn.Module):

    def __init__(self, encoder, encoder_args={}, method='km', temp=10., temp_learnable=True):
        super().__init__()
        self.encoder = models.make(encoder, **encoder_args)
        self.method = method

        if temp_learnable:
            self.temp = nn.Parameter(torch.tensor(temp))
        else:
            self.temp = temp

    def forward(self, x_shot, x_query):
        shot_shape = x_shot.shape[:-3]
        query_shape = x_query.shape[:-3]
        img_shape = x_shot.shape[-3:]

        x_shot = x_shot.view(-1, *img_shape)
        x_query = x_query.view(-1, *img_shape)
        x_tot = self.encoder(torch.cat([x_shot, x_query], dim=0))
        x_shot, x_query = x_tot[:len(x_shot)], x_tot[-len(x_query):]
        x_shot = x_shot.view(*shot_shape, -1)
        x_query = x_query.view(*query_shape, -1)

        if self.method == 'cos':
            # # 方法一：归纳式的原型计算方法
            #x_shot = x_shot.mean(dim=-2)
            # 方法二：我的方法替代了原来的简单求均值
            x_shot = computeProto(x_shot, x_query)
            #x_shot = x_shot.mean(dim=-2)
            
            x_shot = F.normalize(x_shot, dim=-1)
            x_query = F.normalize(x_query, dim=-1)
            metric = 'dot'
        elif self.method == 'sqr':
            x_shot = x_shot.mean(dim=-2)
            metric = 'sqr'
        elif self.method == 'km':
            #x_shot = x_shot.mean(dim=-2)
            x_shot = computeProto(x_shot, x_query)
            x_shot = F.normalize(x_shot, dim=-1)
            x_query = F.normalize(x_query, dim=-1)
            metric = 'km'

        logits = utils.compute_logits(
                x_query, x_shot, metric=metric, temp=self.temp)
        return logits

