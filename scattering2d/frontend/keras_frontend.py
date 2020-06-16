from ...frontend.keras_frontend import ScatteringKeras
from ...scattering2d.frontend.base_frontend import ScatteringBase2D

from ...tensorflow import Scattering2D as ScatteringTensorFlow2D
# 실제로 내가 import 하는건 ?


from tensorflow.python.framework import tensor_shape


class ScatteringKeras2D(ScatteringKeras, ScatteringBase2D): # as Scattering2D (본문에서는)
    def __init__(self, J, L=8, max_order=2, pre_pad=False):
        ScatteringKeras.__init__(self)
        ScatteringBase2D.__init__(self, J, None, L, max_order, pre_pad,'array')
        # ScatteringKeras 와 ScatteringBase2D 의 attribute를 가져옴

    def build(self, input_shape):
        shape = tuple(tensor_shape.TensorShape(input_shape).as_list()[-2:])
        # batch_size를 제외한 나머지를 shape로 정의.

        ########################################################################
        ########################### 여기가 가장 중요 ############################
        ########여기서 정의하는 ScatteringTensorFlow2D 는 class로
        # Tensorflow_frontend.py에서 정의됨. #
        # 이안에는 scattering coefficient를 return 하는 variable S가 있다.
       
        self.S = ScatteringTensorFlow2D(J=self.J, shape=shape,
                                        L=self.L, max_order=self.max_order,
                                        pre_pad=self.pre_pad) #class

        ########################################################################

        # 여기서 2D Scattering을 Tensorflow code에 맞춰서 진행하는 듯

        ScatteringKeras.build(self, input_shape) #keras에서 Layer를 받아서 Layer를 build 한다.
        # NOTE: Do not call ScatteringBase2D.build here since that will
        # recreate the filters already in self.S
        # 번역 : 이미 여기 있는 build함수에서 filters를 만들었으므로, Scattering2D.bulid를 부를
        # 필요 없다.

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        m0 = input_shape[-2] // 2 ** self.J
        m1 = input_shape[-1] // 2 ** self.J
        nc = (self.L ** 2) * self.J * (self.J - 1) // 2 + self.L * self.J + 1
        output_shape = [input_shape[0], nc, m0, m1]
        return tensor_shape.TensorShape(output_shape)


ScatteringKeras2D._document()
