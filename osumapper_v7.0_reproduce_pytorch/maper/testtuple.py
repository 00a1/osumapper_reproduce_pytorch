def ggggg():
    return 1, 2, 3

o = ggggg()

# print(o[2])


import torch
from torch import tensor

# hhh = (tensor([[-0.0795, -0.0170, -0.0152,  ..., -0.0889,  0.0006, -0.0388],
#         [-0.0721, -0.0183, -0.0153,  ..., -0.0879, -0.0001, -0.0446],
#         [-0.0714, -0.0359, -0.0233,  ..., -0.0808,  0.0050, -0.0357],
#         ...,
#         [-0.0823, -0.0311, -0.0034,  ..., -0.0845,  0.0071, -0.0449],
#         [-0.0661, -0.0256, -0.0068,  ..., -0.0883,  0.0072, -0.0472],
#         [-0.0686, -0.0268, -0.0110,  ..., -0.0950, -0.0032, -0.0414]],
#        ), tensor([[[ 0.2946,  0.6321, -0.8698,  0.4934,  0.2946,  0.6321],
#          [ 0.2652,  0.3746, -0.1506, -0.9886,  0.2652,  0.3746],
#          [ 0.1417,  0.5764, -0.6323,  0.7747,  0.1417,  0.5764],
#          ...,
#          [ 0.4492,  0.2834, -0.9267,  0.3759,  0.4492,  0.2834],
#          [ 0.3504,  0.0588, -0.5059, -0.8626,  0.3504,  0.0588],
#          [ 0.1553,  0.0696, -0.9991, -0.0413,  0.1553,  0.0696]],

#         [[ 0.2998,  0.6436, -0.8433,  0.5375,  0.2998,  0.6436],
#          [ 0.2619,  0.3881, -0.1939, -0.9810,  0.2619,  0.3881],
#          [ 0.1633,  0.6129, -0.5046,  0.8633,  0.1633,  0.6129],
#          ...,
#          [ 0.4591,  0.3233, -0.9345,  0.3561,  0.4591,  0.3233],
#          [ 0.3672,  0.0935, -0.4706, -0.8824,  0.3672,  0.0935],
#          [ 0.1723,  0.1107, -0.9978,  0.0658,  0.1723,  0.1107]],

#         [[ 0.2919,  0.6256, -0.8834,  0.4686,  0.2919,  0.6256],
#          [ 0.2243,  0.3813, -0.3462, -0.9382,  0.2243,  0.3813],
#          [ 0.1106,  0.5930, -0.5824,  0.8129,  0.1106,  0.5930],
#          ...,
#          [ 0.4921,  0.3189, -0.8115,  0.5843,  0.4921,  0.3189],
#          [ 0.3837,  0.1022, -0.5546, -0.8321,  0.3837,  0.1022],
#          [ 0.1886,  0.1139, -0.9990, -0.0449,  0.1886,  0.1139]],

#         ...,

#         [[ 0.2881,  0.6155, -0.9030,  0.4296,  0.2881,  0.6155],
#          [ 0.2320,  0.3660, -0.2871, -0.9579,  0.2320,  0.3660],
#          [ 0.2119,  0.6251, -0.1029,  0.9947,  0.2119,  0.6251],
#          ...,
#          [ 0.4776,  0.3525, -0.9338,  0.3578,  0.4776,  0.3525],
#          [ 0.3971,  0.1152, -0.4122, -0.9111,  0.3971,  0.1152],
#          [ 0.2063,  0.1704, -0.9773,  0.2120,  0.2063,  0.1704]],

#         [[ 0.2937,  0.6300, -0.8743,  0.4853,  0.2937,  0.6300],
#          [ 0.2443,  0.3780, -0.2532, -0.9674,  0.2443,  0.3780],
#          [ 0.1952,  0.6301, -0.2509,  0.9680,  0.1952,  0.6301],
#          ...,
#          [ 0.4755,  0.3612, -0.8618,  0.5072,  0.4755,  0.3612],
#          [ 0.3935,  0.1248, -0.4199, -0.9076,  0.3935,  0.1248],
#          [ 0.2012,  0.1707, -0.9844,  0.1760,  0.2012,  0.1707]],

#         [[ 0.2989,  0.6417, -0.8477,  0.5305,  0.2989,  0.6417],
#          [ 0.2498,  0.3897, -0.2513, -0.9679,  0.2498,  0.3897],
#          [ 0.1679,  0.6261, -0.4194,  0.9078,  0.1679,  0.6261],
#          ...,
#          [ 0.4723,  0.3236, -0.8579,  0.5138,  0.4723,  0.3236],
#          [ 0.3710,  0.1010, -0.5186, -0.8550,  0.3710,  0.1010],
#          [ 0.1757,  0.1040, -0.9999, -0.0118,  0.1757,  0.1040]]],
#        ), tensor([[[0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000]],

#         [[0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000]],

#         [[0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000]],

#         [[0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000]],

#         [[0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000]],

#         [[0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000]],

#         [[0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000]],

#         [[0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000]],

#         [[0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000]],

#         [[0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000]],

#         [[0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000]],

#         [[0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000]],

#         [[0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000]],

#         [[0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000]],

#         [[0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000]],

#         [[0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000]],

#         [[0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000]],

#         [[0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000]],

#         [[0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000]],

#         [[0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000]],

#         [[0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000]],

#         [[0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000]],

#         [[0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000]],

#         [[0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000]],

#         [[0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000]],

#         [[0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000]],

#         [[0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000]],

#         [[0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000]],

#         [[0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000]],

#         [[0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000]],

#         [[0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000]],

#         [[0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000]],

#         [[0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000]],

#         [[0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000]],

#         [[0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000]],

#         [[0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000]],

#         [[0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000]],

#         [[0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000]],

#         [[0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000]],

#         [[0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000]],

#         [[0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000]],

#         [[0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000]],

#         [[0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000]],

#         [[0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000]],

#         [[0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000]],

#         [[0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000]],

#         [[0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000]],

#         [[0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000]],

#         [[0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000]],

#         [[0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000],
#          [0.5000]]], ))


sm = tensor([[[0.5000],
         [0.5000],
         [0.5000],
         [0.5000],
         [0.5000],
         [0.5000],
         [0.5000],
         [0.5000],
         [0.5000],
         [0.5000]],

        [[0.5000],
         [0.5000],
         [0.5000],
         [0.5000],
         [0.5000],
         [0.5000],
         [0.5000],
         [0.5000],
         [0.5000],
         [0.5000]],

        [[0.5000],
         [0.5000],
         [0.5000],
         [0.5000],
         [0.5000],
         [0.5000],
         [0.5000],
         [0.5000],
         [0.5000],
         [0.5000]],

        [[0.5000],
         [0.5000],
         [0.5000],
         [0.5000],
         [0.5000],
         [0.5000],
         [0.5000],
         [0.5000],
         [0.5000],
         [0.5000]],

        [[0.5000],
         [0.5000],
         [0.5000],
         [0.5000],
         [0.5000],
         [0.5000],
         [0.5000],
         [0.5000],
         [0.5000],
         [0.5000]], ])

def GenerativeCustomLoss(y_pred):
    classification = y_pred
    if classification.dim() == 1:
        classification = classification.unsqueeze(0)  # Convert to a 2D tensor if it's 1D
    loss1 = 1 - torch.mean(classification, dim=1)
    return loss1


f = tensor([[0.6000], [1.0000], [0.0000], [1.0000], [0.2700], [0.5000], [0.5000], [0.5000], [0.5000], [0.5000], [0.5000], [0.5000], [0.5000], [0.5000], [0.5000], [0.5000], [0.5000], [0.5000], [0.5000], [0.5000]])

# print(hhh[2])
# print(GenerativeCustomLoss(f) * 1)




MappingLayeroutsmall = tensor([[[ 0.2946,  0.6321, -0.8698,  0.4934,  0.2946,  0.6321],
         [ 0.2652,  0.3746, -0.1506, -0.9886,  0.2652,  0.3746],
         [ 0.1417,  0.5764, -0.6323,  0.7747,  0.1417,  0.5764],
         [ 0.4492,  0.2834, -0.9267,  0.3759,  0.4492,  0.2834],
         [ 0.3504,  0.0588, -0.5059, -0.8626,  0.3504,  0.0588],
         [ 0.1553,  0.0696, -0.9991, -0.0413,  0.1553,  0.0696]],

        [[ 0.2998,  0.6436, -0.8433,  0.5375,  0.2998,  0.6436],
         [ 0.2619,  0.3881, -0.1939, -0.9810,  0.2619,  0.3881],
         [ 0.1633,  0.6129, -0.5046,  0.8633,  0.1633,  0.6129],
         [ 0.4591,  0.3233, -0.9345,  0.3561,  0.4591,  0.3233],
         [ 0.3672,  0.0935, -0.4706, -0.8824,  0.3672,  0.0935],
         [ 0.1723,  0.1107, -0.9978,  0.0658,  0.1723,  0.1107]],

        [[ 0.2919,  0.6256, -0.8834,  0.4686,  0.2919,  0.6256],
         [ 0.2243,  0.3813, -0.3462, -0.9382,  0.2243,  0.3813],
         [ 0.1106,  0.5930, -0.5824,  0.8129,  0.1106,  0.5930],
         [ 0.4921,  0.3189, -0.8115,  0.5843,  0.4921,  0.3189],
         [ 0.3837,  0.1022, -0.5546, -0.8321,  0.3837,  0.1022],
         [ 0.1886,  0.1139, -0.9990, -0.0449,  0.1886,  0.1139]],

        [[ 0.2881,  0.6155, -0.9030,  0.4296,  0.2881,  0.6155],
         [ 0.2320,  0.3660, -0.2871, -0.9579,  0.2320,  0.3660],
         [ 0.2119,  0.6251, -0.1029,  0.9947,  0.2119,  0.6251],
         [ 0.4776,  0.3525, -0.9338,  0.3578,  0.4776,  0.3525],
         [ 0.3971,  0.1152, -0.4122, -0.9111,  0.3971,  0.1152],
         [ 0.2063,  0.1704, -0.9773,  0.2120,  0.2063,  0.1704]],

        [[ 0.2937,  0.6300, -0.8743,  0.4853,  0.2937,  0.6300],
         [ 0.2443,  0.3780, -0.2532, -0.9674,  0.2443,  0.3780],
         [ 0.1952,  0.6301, -0.2509,  0.9680,  0.1952,  0.6301],
         [ 0.4755,  0.3612, -0.8618,  0.5072,  0.4755,  0.3612],
         [ 0.3935,  0.1248, -0.4199, -0.9076,  0.3935,  0.1248],
         [ 0.2012,  0.1707, -0.9844,  0.1760,  0.2012,  0.1707]],

        [[ 0.2989,  0.6417, -0.8477,  0.5305,  0.2989,  0.6417],
         [ 0.2498,  0.3897, -0.2513, -0.9679,  0.2498,  0.3897],
         [ 0.1679,  0.6261, -0.4194,  0.9078,  0.1679,  0.6261],
         [ 0.4723,  0.3236, -0.8579,  0.5138,  0.4723,  0.3236],
         [ 0.3710,  0.1010, -0.5186, -0.8550,  0.3710,  0.1010],
         [ 0.1757,  0.1040, -0.9999, -0.0118,  0.1757,  0.1040]]],)

def inblock_loss(vg, border, value):
    wall_var_l = torch.where(vg < border, (value - vg)**2, torch.zeros_like(vg))
    wall_var_r = torch.where(vg > 1 - border, (vg - (1 - value))**2, torch.zeros_like(vg))
    return torch.mean(wall_var_l + wall_var_r)


# box_loss_border 0.1
# box_loss_value 0.4
# box_loss_weight 1
def BoxCustomLoss(loss_border, loss_value, y_pred):
    map_part = y_pred
    return inblock_loss(map_part[0:2], loss_border, loss_value) + inblock_loss(map_part[4:6], loss_border, loss_value)

# print(BoxCustomLoss(0.1, 0.4, MappingLayeroutsmall) * 1)
# print(BoxCustomLoss(0.1, 0.4, MappingLayeroutsmall))

# print(torch.tensor(0.0, dtype=torch.float32) * 1e-8)


print(torch.tensor(0.0, dtype=torch.float32) * 1e-8 + BoxCustomLoss(0.1, 0.4, MappingLayeroutsmall) * 1 + GenerativeCustomLoss(f) * 1)