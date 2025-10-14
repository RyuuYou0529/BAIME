from ..core.register import INFERENCER_REGISTER

@INFERENCER_REGISTER.register('BaseInferencer')
class BaseInferencer(object):
    def __init__(self, args=None):
        super(BaseInferencer, self).__init__()
    
    def infer(self):
        pass
