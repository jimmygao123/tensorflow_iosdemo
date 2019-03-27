import tfcoreml as tf_converter

tf_converter.convert(tf_model_path='mnist.pb',
                     mlmodel_path='my_model.mlmodel',
                     output_feature_names=['Softmax:0'],input_name_shape_dict={"input/x_input:0":[1,784]})