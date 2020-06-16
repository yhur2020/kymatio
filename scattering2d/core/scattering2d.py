# Authors: Edouard Oyallon, Muawiz Chaudhary
# Scientific Ancestry: Edouard Oyallon, Laurent Sifre, Joan Bruna
import tensorflow as tf
import numpy as np

def scattering2d(x, pad, unpad, backend, J, L, phi, psi, max_order, out_type='array'):
    # keras 로 실행하면 backend는 tensorflow_frontend 에서 'tensorflow'임.
    # 여기서 받는 x은 batch size가 포함된 image (?, m,n)
    subsample_fourier = backend.subsample_fourier
    modulus = backend.modulus
    fft = backend.fft
    cdgmm = backend.cdgmm
    concatenate = backend.concatenate

    # Define lists for output.
    out_S_0, out_S_1, out_S_2 = [], [], []

    # 0.
    U_r = pad(x)
    # [[6, 6, 6, 6]] reflection padding for MNIST 
    # output image shape (?, 40, 40, 1)
 
    ### Pooling_layer_0 ###############################################################
    # U_r_m = tf.expand_dims(U_r,3)
    # U_r_m = tf.nn.max_pool2d(tf.cast(U_r_m, dtype = tf.float64) , ksize=(1,2,2,1), strides = (1,2,2,1), padding='VALID')
    # Max-pooling with data type : tf.float64
    # Because tf.nn.max_pool is "invalid"for the complex type data.
    # U_r = tf.reshape(U_r_m, tf.shape(U_r_m)[:-1])
    # U_r = tf.cast(U_r, dtype= tf.complex64)
    # Again cast the tensor to tf.complex64 for fft.
    ###################################################################################

    U_0_c = fft(U_r, 'C2C') # tf.signal.fft2d with input size : (?,m_padded/2,n_padded/2)
                            
    # ouput size : (?,40/2,40/2) afte max-pooling. (ver.1)
    # ouput size : (?,40,40)

    # First low pass
    U_1_c = cdgmm(U_0_c, phi[1][0]) # pointwise tensor product
                                 # i.e. convolution in spatial domain
                                 # Here phi[0][0] is cropped version of filter of size (40,40)
                                 # and if max-pooling works, the filter size should be (20,20)
                                 # constructed in 'filter_bank.py'
                                 #  here size of phi[0][0] : 20 x 20.
                                 # 


    U_1_c = subsample_fourier(U_1_c, k=2 ** J) # 기존 k= 2**J
        # shape : (?, 20, 20) -> (?, 4,5,4,5) -> (?,5,5) 
        # 첫번째는 reshape, 두번째는 axis = (1,3)에 대한 reduce_mean
        # 
        # F^{-1}(out)[u1, u2] = F^{-1}(x)[u1 * 2^J, u2 * 2^J] 
        
        # 각 axis에 20개씩 있던 value를 5개로 줄여서 sampling함
        # 
    S_0 = fft(U_1_c, 'C2R', inverse=True) # inverse fft
    # output size : (?, 5, 5)
    

    S_0 = unpad(S_0)
    
    
    

    out_S_0.append({'coef': S_0,
                    'j': (),
                    'theta': ()}) # S_0 에 append 하는데 이때 dictionary로 append
                                  # j 와 theta에는 값 할당 없음. psi를 사용하지 않았기 때문.


    # 1
    for n1 in range(len(psi)):  #n1 : 0 ~ 7 
                                ## -> j = 0에 해당하는 것 여기서는 각각 한개씩만 psi가 들어 있음 

                                #n1 : 8 ~ 15 
                                ## -> j = 1에 해당하는 것 여기서는 psi가 (같은 psi에 대해) 두개씩 들어 있음
                                ##    
                                
                                #n1 : 16~23 
                                ## -> j = 2 에 해당하는 것 psi가 두개씩 들어 있음
                                ##    
                                ##    같은 filter임에도 layer가 달라짐에 따라 support 가 달라지기 때문에 이렇게 저장해둠.
        j1 = psi[n1]['j']
        theta1 = psi[n1]['theta']

        U_1_c = cdgmm(U_0_c, psi[n1][0]) # 40 x 40 
        
        #########################
        if j1 > 0 : 
            U_1_c = subsample_fourier(  U_1_c, k=2 **(j1)  ) # [40/2^(j1+1)] x [40/2^(j1+1)]
        #########################

        U_1_c = fft(U_1_c, 'C2C', inverse=True) 
        U_1_c = modulus(U_1_c) # [40/2^(j1 + 1)]  x  [40/2^(j1 + 1)]
                               # 만약 j1 이 2이면, 5  x  5 image
                               # 만약 j1 이 1이면, 10 x 10 image
                               # 만약 j1 이 0이면, 20 x 20 image
        ### Pooling_layer_1 ###############################################################
        # expand dim for the max_pooling
        U_1_c_m = tf.expand_dims(U_1_c, 3)

        U_1_c_m = tf.nn.max_pool2d(U_1_c_m, ksize=(1,2,2,1), strides = (1,2,2,1), padding = 'VALID')
        U_1_c = tf.reshape(U_1_c_m, tf.shape(U_1_c_m)[:-1])
        U_1_c = tf.cast(U_1_c, dtype = tf.complex64)
        # cast the tensor to complex data type for the fft.

        ###################################################################################
        # 만약 j1 이 2 이면, max-pooling 되고 난 후는 2  x  2 imgae.
        # 만약 j1 이 1 이면, max-pooling 되고 난 후는 5  x  5 image.
        # 만약 j1 이 0 이면, max-pooling 되고 난 후는 10 x 10 image.
        # [40/2^(j1 + 1)] x [40/2^(j1 + 1)]
        U_1_c = fft(U_1_c, 'C2C') 
        U_1_c = tf.cast(U_1_c, dtype = tf.complex64)       
                                   
        # Second low pass filter
        # ********************************************************
        S_1_c = cdgmm(U_1_c, phi[j1][1]) # using j1 at layer1 
        # ********************************************************

        # print("The value for subsample is J-j1-2 : J-{}-2 is {}".format(j1, J-j1-2))
        
        # print("The shape of S_1_c for j1 : {} is {}".format(j1, S_1_c.shape))
        
        
        # input size : 40/2^{j1 + 1}
        S_1_c = subsample_fourier(S_1_c, k=2 ** (J - j1))
        # output = 40/2^{J+1}
        
        
        S_1_r = fft(S_1_c, 'C2R', inverse=True)
        # S_1_r = unpad(S_1_r)

        out_S_1.append({'coef': S_1_r,
                        'j': (j1,),
                        'theta': (theta1,)})

        # 2
        if max_order < 2:
            continue
        for n2 in range(len(psi)): # For each psi indexed by n2
            j2 = psi[n2]['j']
            theta2 = psi[n2]['theta']

            if j2 <= j1: 
                continue #(j1 < j2)
            


            # input size : [40/2^(j1 + 1)] x [40/2^(j1 + 1)]
            # specially the case j1 = 0 : [40/2^2] x [40/2^2]
            if j1 == 0 :
                U_2_c = cdgmm(U_1_c, psi[n2][2])
                # output size : [40/2^1] x [40/2^1]
            else : # j1 = 1
                U_2_c = cdgmm(U_1_c, psi[n2][1])
            # output size : [40/2^(j1 + 1)] x [40/2^(j1 + 1)]
            
            #print(j1)
            #print(j2)
            #print(U_2_c)
            U_2_c = subsample_fourier(U_2_c, k=2 ** (j2 - j1)) # 40/2^(j2+1) x 40/2^(j2+1)
            U_2_c = fft(U_2_c, 'C2C', inverse=True)
            U_2_c = modulus(U_2_c)
            #print(U_2_c)
            #print("----------------------------------------------------")

            ### Pooling_layer_2 ###############################################################
            # expand dim for the max_pooling
            U_2_c_m = tf.expand_dims(U_2_c, 3)
            U_2_c_m = tf.nn.max_pool2d(U_2_c_m, ksize=(1,2,2,1), strides = (1,2,2,1), padding = 'VALID')
            U_2_c = tf.reshape(U_2_c_m, tf.shape(U_2_c_m)[:-1])
            U_2_c = tf.cast(U_2_c, dtype = tf.complex64)
            # cast the tensor to complex data type for the fft.
            ###################################################################################
            # output size : 40/2^(j2+2) x 40/2^(j2+2)
    

            U_2_c = fft(U_2_c, 'C2C')
            # Third low pass filter


            S_2_c = cdgmm(U_2_c, phi[j2][2])
            
            
            S_2_c = subsample_fourier(S_2_c, k=2 ** (J - j2)) 

            S_2_r = fft(S_2_c, 'C2R', inverse=True) #40/2^J x 40/2^J
            
            # I want skip this part 
            # S_2_r = unpad(S_2_r)
            # output size : 3x3
            

            out_S_2.append({'coef': S_2_r,
                            'j': (j1, j2),
                            'theta': (theta1, theta2)})

    out_S = []
    #print(np.shape(out_S_0))
    out_S.extend(out_S_0)

    #print(np.shape(out_S_1))
    out_S.extend(out_S_1)
    #print(np.shape(out_S_2))
    out_S.extend(out_S_2)
    # out_S is 'list' of dictionaries


    if out_type == 'array':
        for x in out_S :
            # print(x['coef'].get_shape().as_list())
            m = x['coef'].get_shape().as_list()[-2]
            n = x['coef'].get_shape().as_list()[-1]
            x['coef']= tf.reshape(x['coef'], (-1, m*n))
        
        out_S = tf.concat([x['coef'] for x in out_S],axis = 1 )
        print(out_S.get_shape().as_list())

        #tf.reshape([x['coef'] for x in out_S], (-1, len(x['coef'][-2]) * len(x['coef'][-1] )))
        #out_S = concatenate([x['coef'] for x in out_S])
        # backend.concatenate = lambda x: concatenate(x, -3)
    return out_S # image by image stack
    


##################################################################################
##################################################################################    
""" def Maxpooling2dtensor(inputs) :
    
    M, N = tf.shape(inputs)[:-2]
    
    batch_size = tf.shape(inputs)[-2:]
    
    inputs = inputs.eval()

    out = np.zeros([batch_size, M//2,N//2])

    temp = np.zeros([M//2,N//2])
    for k in range(batch_size):
        for i in range(M//2) :
            for j in range(N//2) :
                temp[i,j] = np.amax(inputs[ k ,2*i : 2*i+2 , 2*j : 2*j+2 ])
                out[k,i,j] = temp[i,j]
    out = tf.convert_to_tensor(out, dtype = tf.float32)
    return out
##################################################################################
##################################################################################
def Maxpooling2dtensor_1(inputs) :
    
    M, N = tf.get_shape(inputs)[-2:]
    
    inputs = inputs.eval()

    out = np.zeros([1, M//2,N//2])

    temp = np.zeros([M//2,N//2])
    for i in range(M//2) :
        for j in range(N//2) :
            temp[i,j] = np.amax(inputs[ k ,2*i : 2*i+2 , 2*j : 2*j+2 ])
            out[0,i,j] = temp[i,j]

    out = tf.convert_to_tensor(out, dtype = tf.float32)
    return out
##################################################################################
##################################################################################
"""




__all__ = ['scattering2d']
