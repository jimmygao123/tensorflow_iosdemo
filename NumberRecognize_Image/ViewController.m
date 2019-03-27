//
//  ViewController.m
//  NumberRecognize_Image
//
//  Created by mtgao on 2019/3/26.
//  Copyright © 2019 ggd. All rights reserved.
//

#import "ViewController.h"
#import <CoreML/CoreML.h>
#import "my_model.h"

@interface ViewController ()
@property (weak, nonatomic) IBOutlet UIImageView *imageView;
@property (weak, nonatomic) IBOutlet UILabel *label;

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    [self doPredict];
}

- (void)doPredict{
    //加载图片
    NSString *testImagePath = [[NSBundle mainBundle] pathForResource:@"test_image.png" ofType:nil];
    UIImage *testImage = [[UIImage alloc]initWithContentsOfFile:testImagePath];
    self.imageView.image = testImage;
    
    size_t width = CGImageGetWidth(testImage.CGImage);
//    size_t bytePerRow = CGImageGetBytesPerRow(testImage.CGImage);
    size_t height = CGImageGetHeight(testImage.CGImage);

    //获取图片原始数据
    CGDataProviderRef dataProvider = CGImageGetDataProvider(testImage.CGImage);
    CFDataRef rawData = CGDataProviderCopyData(dataProvider);
    
    UInt8 *testImagedata = (UInt8 *)CFDataGetBytePtr(rawData);
    CFIndex length = CFDataGetLength(rawData);

    //初始化一维数组
    NSError *error = nil;
    MLMultiArray *mlArray = [[MLMultiArray alloc] initWithShape:@[@(width * height)] dataType:MLMultiArrayDataTypeFloat32 error:&error];
    if(error){
        NSLog(@"input data error = %@",error);
    }
    
    //在一维数组中填充图片的R通道数据
    for(CFIndex i=0, index=0; i<length; i=i+4,index++){
        UInt8 r = testImagedata[i];
        Float32 red_float = r/255.0;
        NSNumber *num = [NSNumber numberWithFloat:red_float];
        [mlArray setObject:num atIndexedSubscript:index];
    }
    
    //加载模型，并预测结果
    my_model *model = [[my_model alloc] init];
    my_modelInput * modelInput = [[my_modelInput alloc] initWithInput__x_input__0:mlArray];
    my_modelOutput * modelOutput = [model predictionFromFeatures:modelInput error:nil];

    
    //读取识别结果并显示
    NSMutableString *displayText = [[NSMutableString alloc] init];
    for(CFIndex index=0; index < modelOutput.Softmax__0.count; index ++){
        NSNumber *num = [modelOutput.Softmax__0 objectAtIndexedSubscript:index];
        NSString *tempString = [NSString stringWithFormat:@"%ld: 可能的机率是%f\n",(long)index,num.floatValue];
        [displayText appendString:tempString];
    }
 
    dispatch_async(dispatch_get_main_queue(), ^{
        self.label.text = displayText;
    });
}

@end
