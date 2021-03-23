from .segnet import segnet
from .synnet import synnet
from keras.layers import Input
from keras.models import Model
import sampling

def vicce():
    s_inputs = Input(shape=(256,256,1))
    c_inputs = Input(shape=(256,256,1))
    
    x_s = segnet(s_inputs)
    x_c = segnet(c_inputs)

    x_s = sampling(x_s)
    x_c = sampling(x_c)

    s_outputs = synnet(x_s)
    c_outputs = synnet(x_c)

    model = Model(inputs  = [s_inputs, c_inputs], 
                  outputs = [s_outputs, c_outputs])
    # fit([train_sp, train_cr],[train_cr, train_sp])
    return model