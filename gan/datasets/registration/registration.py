import SimpleITK as sitk
import os
import matplotlib.pyplot as plt

def initial_registration(fixed_image, moving_image, initial_transform):
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=50, estimateLearningRate=registration_method.Once)
    registration_method.SetOptimizerScalesFromPhysicalShift() 
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas = [2,1,0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

#下面三行注释掉的，如果在jupyter里，可以返回训练过程metric的变化曲线

    # registration_method.AddCommand(sitk.sitkStartEvent, rc.metric_start_plot)
    # registration_method.AddCommand(sitk.sitkEndEvent, rc.metric_end_plot)
    # registration_method.AddCommand(sitk.sitkIterationEvent, 
    #                                lambda: rc.metric_plot_values(registration_method))    
    
    
    final_transform = registration_method.Execute(fixed_image, moving_image)
    print('Final metric value: {0}'.format(registration_method.GetMetricValue()))
    print('Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))
#     return (final_transform, registration_method.GetMetricValue())
    return final_transform

def perform_transform(transform, fixed_image, moving_image):
    """
    Write the given transformation to file, resample the moving_image onto the fixed_images grid.
    
    Args:
        transform (SimpleITK Transform): transform that maps points from the fixed image coordinate system to the moving.
        fixed_image (SimpleITK Image): resample onto the spatial grid defined by this image.
        moving_image (SimpleITK Image): resample this image.
    """                             
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(fixed_image)
    # SimpleITK supports several interpolation options, we go with the simplest that gives reasonable results. 
    resample.SetInterpolator(sitk.sitkLinear)  
    resample.SetTransform(transform)
    return resample.Execute(moving_image)

def bspline_registration(fixed_image, moving_image, fixed_image_mask=None, fixed_points=None, moving_points=None):

    registration_method = sitk.ImageRegistrationMethod()
    
    # Determine the number of BSpline control points using the physical spacing we want for the control grid. 
    grid_physical_spacing = [50.0, 50.0, 50.0] # A control point every 50mm
    image_physical_size = [size*spacing for size,spacing in zip(fixed_image.GetSize(), fixed_image.GetSpacing())]
    mesh_size = [int(image_size/grid_spacing + 0.5) \
                 for image_size,grid_spacing in zip(image_physical_size,grid_physical_spacing)]

    initial_transform = sitk.BSplineTransformInitializer(image1 = fixed_image, 
                                                         transformDomainMeshSize = mesh_size, order=3)    
    registration_method.SetInitialTransform(initial_transform)
        
    #registration_method.SetMetricAsMeanSquares()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    # Settings for metric sampling, usage of a mask is optional. When given a mask the sample points will be 
    # generated inside that region. Also, this implicitly speeds things up as the mask is smaller than the
    # whole image.
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    if fixed_image_mask:
        registration_method.SetMetricFixedMask(fixed_image_mask)
    
    # Multi-resolution framework.            
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations= 100)
    
    # registration_method.AddCommand(sitk.sitkStartEvent, rc.metric_start_plot)
    # registration_method.AddCommand(sitk.sitkEndEvent, rc.metric_end_plot)
    # registration_method.AddCommand(sitk.sitkIterationEvent, 
    #                                lambda: rc.metric_plot_values(registration_method))
    
    # If corresponding points in the fixed and moving image are given then we display the similarity metric
    # and the TRE during the registration.
    if fixed_points and moving_points:
        registration_method.AddCommand(sitk.sitkStartEvent, rc.metric_and_reference_start_plot)
        registration_method.AddCommand(sitk.sitkEndEvent, rc.metric_and_reference_end_plot)
        registration_method.AddCommand(sitk.sitkIterationEvent, lambda: rc.metric_and_reference_plot_values(registration_method, fixed_points, moving_points))

    ffd_transform = registration_method.Execute(fixed_image, moving_image)
    metric_value = registration_method.GetMetricValue()
    print('FFD_final metric value: {0}'.format(registration_method.GetMetricValue()))
    print('Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))
#     return registration_method.Execute(fixed_image, moving_image)
    return (ffd_transform, metric_value)

def bspline_registration_morepoint(fixed_image, moving_image, fixed_image_mask=None, fixed_points=None, moving_points=None):

    registration_method = sitk.ImageRegistrationMethod()
    
    # Determine the number of BSpline control points using the physical spacing we 
    # want for the finest resolution control grid. 
    grid_physical_spacing = [10.0, 10.0, 10.0] # A control point every 50mm
    image_physical_size = [size*spacing for size,spacing in zip(fixed_image.GetSize(), fixed_image.GetSpacing())]
    mesh_size = [int(image_size/grid_spacing + 0.5) \
                 for image_size,grid_spacing in zip(image_physical_size,grid_physical_spacing)]

    # The starting mesh size will be 1/4 of the original, it will be refined by 
    # the multi-resolution framework.
    mesh_size = [int(sz/4 + 0.5) for sz in mesh_size]
    
    initial_transform = sitk.BSplineTransformInitializer(image1 = fixed_image, 
                                                         transformDomainMeshSize = mesh_size, order=3)    
    # Instead of the standard SetInitialTransform we use the BSpline specific method which also
    # accepts the scaleFactors parameter to refine the BSpline mesh. In this case we start with 
    # the given mesh_size at the highest pyramid level then we double it in the next lower level and
    # in the full resolution image we use a mesh that is four times the original size.
    registration_method.SetInitialTransformAsBSpline(initial_transform,
                                                     inPlace=True,
                                                     scaleFactors=[1,2,4])
#     registration_method.SetMetricAsMeanSquares()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    # Settings for metric sampling, usage of a mask is optional. When given a mask the sample points will be 
    # generated inside that region. Also, this implicitly speeds things up as the mask is smaller than the
    # whole image.
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    if fixed_image_mask:
        registration_method.SetMetricFixedMask(fixed_image_mask)
    
    # Multi-resolution framework.            
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    registration_method.SetInterpolator(sitk.sitkLinear)
    # Use the LBFGS2 instead of LBFGS. The latter cannot adapt to the changing control grid resolution.
    registration_method.SetOptimizerAsLBFGS2(solutionAccuracy=1e-8, numberOfIterations=100, deltaConvergenceTolerance=1e-8)

    # registration_method.AddCommand(sitk.sitkStartEvent, rc.metric_start_plot)
    # registration_method.AddCommand(sitk.sitkEndEvent, rc.metric_end_plot)
    # registration_method.AddCommand(sitk.sitkIterationEvent, 
    #                                lambda: rc.metric_plot_values(registration_method))    
    
    # If corresponding points in the fixed and moving image are given then we display the similarity metric
    # and the TRE during the registration.
    if fixed_points and moving_points:
        registration_method.AddCommand(sitk.sitkStartEvent, rc.metric_and_reference_start_plot)
        registration_method.AddCommand(sitk.sitkEndEvent, rc.metric_and_reference_end_plot)
        registration_method.AddCommand(sitk.sitkIterationEvent, lambda: rc.metric_and_reference_plot_values(registration_method, fixed_points, moving_points))

    ffd_transform = registration_method.Execute(fixed_image, moving_image)
    metric_value = registration_method.GetMetricValue()
    print('FFD_morepoint_final metric value: {0}'.format(registration_method.GetMetricValue()))
    print('Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))
    return (ffd_transform,metric_value)   

#三步配准并保存配准结果
def registration_three_phase(nii_path, save_folder):
    patient_folders = os.listdir(nii_path)
    patient_folders.sort()
    # patient_folders.reverse()

    #如果想单独配准哪几个病人可以改改下面这个list
    # patient_folders = ['5049830','4230975','3911806']
    print(patient_folders)
    for patient_folder in patient_folders:
        print(patient_folder)
        patient_folder_path = os.path.join(nii_path,patient_folder)
        nii_files = os.listdir(patient_folder_path)
        for nii_file in nii_files:
            if 'CTA' in nii_file:
                cta_path = os.path.join(patient_folder_path,nii_file)
            if 'MBF' in nii_file:
                mbf_path = os.path.join(patient_folder_path,nii_file)
            if 'MIP' in nii_file:
                mip_path = os.path.join(patient_folder_path,nii_file) 
        save_path = os.path.join(save_folder,patient_folder)  
        os.makedirs(save_path, exist_ok=True) 
        fixed_image =  sitk.ReadImage(cta_path, sitk.sitkFloat32)
        moving_image = sitk.ReadImage(mip_path, sitk.sitkFloat32) 
        mbf_image = sitk.ReadImage(mbf_path, sitk.sitkFloat32) 
        print('ok_ready')


        #第一步（种）配准：直接调用ffd配准模型
        direct_ffd_transform,ffd_metric = bspline_registration(fixed_image = fixed_image, 
                                            moving_image = moving_image,
                                            fixed_image_mask = None,
                                            fixed_points = None, 
                                            moving_points = None
                                            )

        mip_ffd = perform_transform(direct_ffd_transform, fixed_image,moving_image)
        mbf_ffd = perform_transform(direct_ffd_transform, fixed_image,mbf_image)

        dst_mip = os.path.join(save_path,patient_folder + '_MIP_regis_direct_ffd_'+str(ffd_metric)+'.nii.gz')
        dst_mbf = os.path.join(save_path,patient_folder + '_MBF_regis_direct_ffd_'+str(ffd_metric)+'.nii.gz')
        sitk.WriteImage(mip_ffd, dst_mip)
        sitk.WriteImage(mbf_ffd, dst_mbf)

        print('ok')
        #第二步（种）配准：ffd with a multi-resolution control point grid 模型配准
        ffd_transform_morepoint,ffd_morepoint_metric = bspline_registration_morepoint(fixed_image = fixed_image, 
        #                                       moving_image = mip_affine,
                                            moving_image = moving_image,
                                            fixed_image_mask = None,
                                            fixed_points = None, 
                                            moving_points = None
                                            )

        mip_ffd_morepoint = perform_transform(ffd_transform_morepoint,fixed_image,moving_image)
        mbf_ffd_morepoint = perform_transform(ffd_transform_morepoint,fixed_image,mbf_image)

        dst_mip2=  os.path.join(save_path,patient_folder +'_MIP_regis_direct_ffd_morepoint_'+str(ffd_morepoint_metric)+'.nii.gz')
        dst_mbf2=  os.path.join(save_path,patient_folder + '_MBF_regis_direct_ffd_morepoint_'+str(ffd_morepoint_metric)+'.nii.gz')
        sitk.WriteImage(mip_ffd_morepoint, dst_mip2)
        sitk.WriteImage(mbf_ffd_morepoint, dst_mbf2)

        print('ok')
        #第三步（种）配准模型：第一步的配准结果作为第二步模型的初始值，执行配准
        ffd_ffd_transform_morepoint,ffd_ffd_metric = bspline_registration_morepoint(fixed_image = fixed_image, 
                                            moving_image = mip_ffd,
                                            fixed_image_mask = None,
                                            fixed_points = None, 
                                            moving_points = None
                                            )
        mip_ffd_ffd = perform_transform(ffd_ffd_transform_morepoint,fixed_image,mip_ffd)
        mbf_ffd_ffd = perform_transform(ffd_ffd_transform_morepoint,fixed_image,mbf_ffd)

        dst_mip3=  os.path.join(save_path,patient_folder +'_MIP_regis_ffd_ffd_morepoint_'+str(ffd_ffd_metric)+'.nii.gz')
        dst_mbf3=  os.path.join(save_path,patient_folder + '_MBF_regis_ffd_ffd_morepoint_'+str(ffd_ffd_metric)+'.nii.gz')
        sitk.WriteImage(mip_ffd_ffd, dst_mip3)
        sitk.WriteImage(mbf_ffd_ffd, dst_mbf3)

        print('ok')

if __name__ =='__main__':

    nii_path = '/home/proxima-sx12/dataset/cta_mip_rasample'
    save_folder = '/home/proxima-sx12/regis_results'
    registration_three_phase(nii_path, save_folder)
    #接下来从save_folder中选择配准metrics高的去切片训练
