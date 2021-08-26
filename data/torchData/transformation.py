import torch

class Transform(object):
    """
    CustomTransform

    Level 1: ReduceRes()
    Level 2: CustomUnsqueezeChannel(),AddChannel(),StackChannel()
    Level 3: ToStackImg()
    """

    def __init__(self):
        pass

    def __call__(self, X):
        return torch.Tensor(X)

# Lv1
class ReduceRes(Transform):
    """
    **Custom** Reduce time resolution by factor of 4
    """
    def __call__(self, X):
        return X[:,::4]

class CutFrame(Transform):
    """
    **Custom** Reduce time resolution by factor of 4
    """
    def __init__(self,keep='Amp'):
        self.idx = 70
        if keep = 'Amp':
            self.idx = 70
        elif keep = 'Phase'
            self.idx = 0
        else:
            raise ValueError("Must be either 'Amp' or 'Phase'")

    def __call__(self, X):
        return X[self.idx:self.idx+70,:]

# Lv2
class Unsqueeze(Transform):
    """
    unsqueeze channel

    Example:
    input: tensor with shape (70,1600)
    dim: 0
    return: tensor with shape (1,70,1600)
    """
    def __init__(self,dim=0):
        self.dim = dim

    def __call__(self, X):
        return X.unsqueeze(self.dim)

class StackChannel(Transform):
    """
    Make copy of the image and stack along the channel to increase the number of channel

    Example:
    input: tensor with shape (1,70,1600)
    stack: 3
    dim: 0
    return: tensor with shape (3,70,1600)
    """
    def __init__(self,stack=3,dim=0):
        self.stack = stack
        self.dim = dim

    def __call__(self, X):
        assert len(X.shape) == 3 and X.shape[0] == 1, f'torchsize must be (1,w,l), current size: {X.shape}'
        return torch.cat([X for _ in range(self.stack)],dim=self.dim)

class UnsqueezebyRearrange(Transform):
    """
    **Custom** transform 1-channel Amptitude-PhaseShift frame into 2 channel frame, with Amptitude on top of PhaseShift

    Example:
    input: tensor with shape (1,140,1600)
    return: tensor with shape (2,70,1600)
    """
    def __call__(self, X):
        r,w = X.shape
        return X.reshape(2,r//2,w)

# Lv3
class ToStackImg(Transform):
    """
    Transform a 1 long-frame into numbers of small-frames

    Arugments:
    n_seq:  number of frames to be return

    Example:
    input: tensor with shape (1,70,1600)
    n_seq: 10
    return: tensor with shape (10,1,70,160)
    """
    def __init__(self,n_seq):
        self.n_seq = n_seq

    def __call__(self, X):
        c,r,w = X.shape
        assert w%self.n_seq == 0, 'length must be able to be divided by n_seq'
        return X.reshape(c,r,self.n_seq,w//self.n_seq).permute(2,0,1,3)
