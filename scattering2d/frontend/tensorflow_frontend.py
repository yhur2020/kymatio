import tensorflow as tf
from ...frontend.tensorflow_frontend import ScatteringTensorFlow
from ...scattering2d.core.scattering2d import scattering2d
from .base_frontend import ScatteringBase2D



class ScatteringTensorFlow2D(ScatteringTensorFlow, ScatteringBase2D):

    def __init__(self, J, shape, L=8, max_order=2, pre_pad=False,
            backend='tensorflow', name='Scattering2D', out_type='array'):
        # 여기서의 shape는 batch size 제외 순수 image의 size
        ScatteringTensorFlow.__init__(self, name)
        # -> ScatteringTensorFlow 클래스의 속성(attribute) 중 name을 사용하겠다.
        # Super().__init__() 대신에 superclass의 이름으로도 가능하다.
        
        ###################################################################################
        ###################################################################################
        ScatteringBase2D.__init__(self, J, shape, L, max_order, pre_pad,
                backend, out_type)
        # -> ScatteringBase2D 클래스의 속성(attribute)를 사용하겠다.
        # -> J, shape 같은 argument를 받아서 scattering을 위한 기본적인 세팅을 해줌



        ScatteringBase2D._instantiate_backend(self, 'kymatio.scattering2d.backend.')
        # 여기서 어떤 backend로 scattering을 할지를 결정.
        # 이때 _instantiate_backend가 ScatteringBase2D 에는 없고 상위 클래스인 ScatteringBase
        # 에 존재하기 때문에 잘 안보임.

        ScatteringBase2D.build(self)
        # build는 input의 size와 패딩과 관련된 메서드


        ScatteringBase2D.create_filters(self)
        # -> output : self.phi, self.psi = filters['phi'], filters['psi']
        
        ####################################################################################
        ####################################################################################

        # __init__ 는 이 class가 만들어지면 자동으로 실행됨

    def scattering(self, input): # 여기서 받는 image는 (?, m,n) 으로 batch size 포함이다.
        with tf.name_scope('scattering') as scope:
            try:
                input = tf.convert_to_tensor(input)
            except ValueError:
                raise TypeError('The input should be convertible to a '
                                'TensorFlow Tensor.')

            if len(input.shape) < 2:
                raise RuntimeError('Input tensor should have at least two '
                                   'dimensions.')

            if (input.shape[-1] != self.N or input.shape[-2] != self.M) and not self.pre_pad:
                raise RuntimeError('Tensor must be of spatial size (%i,%i).' % (self.M, self.N))

            if (input.shape[-1] != self.N_padded or input.shape[-2] != self.M_padded) and self.pre_pad:
                raise RuntimeError('Padded tensor must be of spatial size (%i,%i).' % (self.M_padded, self.N_padded))
            if not self.out_type in ('array', 'list'):
                raise RuntimeError("The out_type must be one of 'array' or 'list'.")

            # Use tf.shape to get the dynamic shape
            # execution time.
            batch_shape = tf.shape(input)[:-2] 
            signal_shape = tf.shape(input)[-2:]
            # This assume that the input's channel is 1 implicitly. 
            # In other words, input size = (?, M, N)
            # so that tf.shape(input)[-2:] = (M, N)
            # and tf.shape(input)[:-2] = (?) the batch size.

            # NOTE: Cannot simply concatenate these using + since they are
            # tf.Tensors and that would add their values.
            
            input = tf.reshape(input, tf.concat( ( (-1,), signal_shape), 0 ) )
            # setting the input shape
            # reshaping the input by (?,M,N,) 
            
        
            ####################################################################################
            ####################################################################################
            S = scattering2d(input, self.pad, self.unpad, self.backend, self.J, self.L, self.phi, self.psi,
                             self.max_order, self.out_type)
                             # self.backend  = 'tensorflow'
            ####################################################################################
            ####################################################################################
            

            if self.out_type == 'array':
                scattering_shape = tf.shape(S)[-1:]
                new_shape = tf.concat((batch_shape, scattering_shape), 0)
                S = tf.reshape(S, new_shape) # new_shape

                
            else:
                scattering_shape = tf.shape(S[0]['coef'])[-2:]
                new_shape = tf.concat((batch_shape, scattering_shape), 0)

                for x in S:
                    x['coef'] = tf.reshape(x['coef'], new_shape)

            return S


ScatteringTensorFlow2D._document()


__all__ = ['ScatteringTensorFlow2D']
