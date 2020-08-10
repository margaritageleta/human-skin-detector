# Human Skin Segmentation for Hand Gesture classification
> With thresholding based on YCbCr, HSV, RBG and CIEL*a*b* color spaces intersection and morphological operations in gray scale

Skin segmentation is basic and key in the hand gesture recognition. Since skin color is within a threshold range, thresholding a color space, we can swiftly segment the desired region. However, there are two factors that have a direct influence upon this whole process:

+ Illumination conditions, noise effects and complex backgrounds.
+ Using a particular color space can influence image information representation.

In this paper we propose two alternatives to approach the segmentation problem: 
+ histogram-based and 
+ color space heuristics.

After successfully segmenting human skin, we train a neural network for hand gesture detection to predict how many fingers are represented by the gesture and the performance is assessed with several metrics with respect to the segmentation hyperparameters.
