from __future__ import print_function
 
import SimpleITK as sitk


def N4BiasFieldCorrection(image):
    
    inputImage = sitk.GetImageFromArray(image)
    # Creation mask with otsu threshold
    maskImage = sitk.OtsuThreshold(inputImage, 0, 1,200)
    
    inputImage = sitk.Cast( inputImage, sitk.sitkFloat32 )
    
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    
    output = corrector.Execute(inputImage, maskImage)    
    
    return output