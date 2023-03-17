########################################################
#python train.py --exp_id 3
# weights [1.0, 1.0, 1.0, 1.0] L1 Loss

#python train.py --exp_id 4
# weights [0.1, 1.0, 0.1, 1.0]

#python train.py --exp_id 5
# weights [1.0, 1.0, 1.0, 1.0] joint 3d nn.SmoothL1

#python train.py --exp_id 6
# weights [0.1, 1.0, 0.1, 1.0] joint 3d nn.SmoothL1

#python train.py --exp_id 8
# weights [0.0, 1.0, 0.1, 1.0] joint 3d nn.SmoothL1, no mask loss

#python train.py --exp_id 9
# weights [0.1, 1.0, 0.1, 1.0] joint 3d nn.SmoothL1, 引入了Res6D的技术，都是在小批量数据集上

#python train.py --exp_id 10
# weights [0.1, 1.0, 1, 1.0] joint 3d nn.SmoothL1, 引入了Res6D的技术，都是在小批量数据集上

#python train.py --exp_id 11
# weights [0.1, 1.0, 0.05, 1.0] joint 3d nn.SmoothL1, 引入了Res6D的技术，都是在小批量数据集上

#python train.py --exp_id 12
# weights [0.1, 1.0, 0.05, 0.5] joint 3d nn.SmoothL1, 引入了Res6D的技术，都是在小批量数据集上

#python train.py --exp_id 13
# weights [0.1, 1.0, 0.1, 1.0] joint 3d nn.SmoothL1, 引入了Res6D的技术，都是在小批量数据集上,然后推出得到的delta t之后，3D点用绝对值

#python train.py --exp_id 14
# weights [0.1, 1.0, 0.1, 1.0] joint 3d nn.SmoothL1, 引入了Res6D的技术，都是在小批量数据集上,然后推出得到的delta t之后，3D点d * (delta x + x_0 / d_0)

#python train.py --exp_id 15
# weights [0.1, 1.0, 0.1, 1.0] joint 3d nn.L1, 在大批量数据集上train和test, 100epoch, L1 loss

#python train.py --exp_id 16
# weights [0.1, 1.0, 0.1, 1.0] joint 3d nn.L1, 引入了Res6D的技术，都是在大批量数据集上,然后推出得到的delta t之后，3D点d * (delta x + x_0 / d_0)，100epoch, L1 loss

#python train.py --exp_id 17
# weights [0.1, 1.0, 0.1, 1.0] joint 3d nn.L1, 都是在大批量数据集上,然后推出得到的delta t之后，3D点d * (delta x + x_0 / d_0)，500epoch, L1 loss