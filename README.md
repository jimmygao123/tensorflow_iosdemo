

###  tensorflow_iosdemo

A Objective-C versioin of https://github.com/taoweiji/TensorflowIOSDemo.git

a demo for image number reconginize implemented by CoreML framework.

the model is trained by tensorflow.



####How the model file generated.

1. Tensorflow Enviroment is needed, then use saveModel.py to download dataset, train the model, then save as *.pb format file.
2. use testModel.py to verify whether the model is correct.
3. use convert_model.py to convert *.pb file to *.mlmodel which is apple CoreML supported.
4. Drag the *.mlmodel file to Xcode.