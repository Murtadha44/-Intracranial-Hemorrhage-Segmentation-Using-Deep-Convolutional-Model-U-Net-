import numpy as np, os, pickle, cv2, glob
from imageio import imread
from keras.callbacks import ModelCheckpoint
from sklearn import metrics
from imageio import imsave
from pathlib import Path



from prepare_data import *
from data_process import *
from model import *


def Sens(y_true, y_pred):
    cm1 = metrics.confusion_matrix(y_true, y_pred, labels=[1, 0])  # labels =[1,0] [positive [Hemorrhage], negative]
    SensI = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
    return SensI  # TPR is also known as sensitivity

def Speci(y_true, y_pred):
    cm1 = metrics.confusion_matrix(y_true, y_pred, labels=[1, 0])  # labels =[1,0] [positive [Hemorrhage], negative]
    SpeciI = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
    return SpeciI  # FPR is one minus the specificity or true negative rate

def Jaccard_img(y_true, y_pred): #https://www.jeremyjordan.me/evaluating-image-segmentation-models/
    iou_score=0
    counter=0
    for i in range(y_true.shape[0]):
        if np.sum(y_true[i])>0:#Considering only the slices that have hemorrhage regions, if y_true is all zeros -> iou_score=nan.
            im1 = np.asarray(y_true[i]).astype(np.bool)
            im2 = np.asarray(y_pred[i]).astype(np.bool)
            intersection = np.logical_and(im1, im2)
            union = np.logical_or(im1, im2)
            iou_score+= np.sum(intersection) / np.sum(union)
            counter+=1
    if counter>0:
        return iou_score/counter
    else:
        return np.nan


def dice_img(y_true, y_pred):
    dice=0
    counter = 0
    for i in range(y_true.shape[0]):
        if np.sum(y_true[i]) > 0:  # Considering only the slices that have hemorrhage regions,
            dice += dice_fun(y_true[i], y_pred[i])
            counter += 1
    if counter>0:
        return dice/counter
    else:
        return np.nan

def dice_fun(im1, im2):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / (im1.sum() + im2.sum())



def testModel(model_path,test_path,save_path):
    modelUnet = unet(pretrained_weights=model_path, input_size =(windowLen,windowLen,1) )
    testGener = testGenerator(test_path, target_size =(windowLen,windowLen,1))
    testPredictions = modelUnet.predict_generator(testGener,n_imagesTest,verbose=1)
    saveResult(test_path,save_path,testPredictions) #sending the test image path so same name will be used for saving masks


data_gen_args = dict(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    #brightness_range=(0,1.5),
    fill_mode="nearest"
    )



if __name__=='__main__':
    #############################################Training Parameters#######################################################
    num_CV=5
    NumEpochs=100
    NumEpochEval=1 #validated the model each NumEpochEval epochs
    batch_size = 32
    learning_rateI = 1e-5
    decayI=learning_rateI/NumEpochs
    detectionSen=20*20# labeling each slice as ICH if hemorrhage is detected in detectionSen pixels
    thresholdI= 0.5
    detectionThreshold=thresholdI*256  #threshold on detection probability
    numSubj = 75
    num_WindowsCT=49
    imageLen = 512
    windowLen = 128
    strideLen = 64
    num_Moves = int(imageLen/strideLen)-1
    window_specs = [40, 120]  # Brain window
    kernel_closing = np.ones((10, 10), np.uint8)
    kernel_opening = np.ones((5, 5), np.uint8)# 5*5 in order not to delete thin hemorrhage

    counterI=1; SaveDir = Path('results_trial'+str(counterI))
    while(os.path.isdir(str(SaveDir))):
        counterI+=1
        SaveDir= Path('results_trial'+str(counterI))
    os.mkdir(str(SaveDir))
    os.mkdir(str(Path(SaveDir,'crops')))
    os.mkdir(str(Path(SaveDir,'fullCT_original')))#Testing without morphological operations
    os.mkdir(str(Path(SaveDir, 'fullCT_morph'+ str(thresholdI)))) #Testing with morphological operations
    print('The results of the training, validation and testing will be saved to:'+ str(SaveDir))



    #############################################Downloading and unzipping the dataset######################################
    dataset_zip_dir='computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.3.1.zip'
    crossvalid_dir='DataV1'
    prepare_data(dataset_zip_dir, crossvalid_dir, numSubj, imageLen, windowLen, strideLen,
                 num_Moves, window_specs) #preparing the data and saving it to ICH_DataSegmentV1.pkl

    # Loading full image mask from the crops predictions
    with open(str(Path(crossvalid_dir,'ICH_DataSegmentV1.pkl')), 'rb') as Dataset1:
        [hemorrhageDiagnosisArray, AllCTscans, testMasks, subject_nums_shaffled] = pickle.load(Dataset1)
    del AllCTscans
    testMasks=np.uint8(testMasks)
    testMasksAvg = np.where(np.sum(np.sum(testMasks, axis=1), axis=1) > detectionSen, 1, 0)  #
    testPredictions=np.zeros((testMasks.shape[0],imageLen,imageLen),dtype=np.uint8) #predicted segmentation

    ############################################Cross-validation############################################################
    print('Starting the cross-validation!!')
    for cvI in range(0,num_CV):
        print("Working on fold #" + str(cvI)+", starting training U-Net")
        SaveDir_crops_cv=Path(SaveDir,'crops','CV'+str(cvI))
        if os.path.isdir(str(SaveDir_crops_cv))==False:
            os.mkdir(str(SaveDir_crops_cv))
        SaveDir_full_cv=Path(SaveDir,'fullCT_original','CV'+str(cvI))
        if os.path.isdir(str(SaveDir_full_cv))==False:
            os.mkdir(str(SaveDir_full_cv))
        SaveDir_cv = Path(SaveDir, 'fullCT_morph' + str(thresholdI), 'CV' + str(cvI))
        if os.path.isdir(str(SaveDir_cv)) == False:
            os.mkdir(str(SaveDir_cv))

        dataDir = Path(crossvalid_dir,'CV'+str(cvI))
        n_imagesTrain=len(glob.glob(os.path.join(str(Path(dataDir,'train','image')), "*.png")))
        n_imagesValidate=len(glob.glob(os.path.join(str(Path(dataDir,'validate','image')), "*.png")))
        n_imagesTest = len(glob.glob(os.path.join(str(Path(dataDir,'test','crops','image')), "*.png")))
        trainGener = trainGenerator(batch_size,str(Path(dataDir,'train')),'image','label',data_gen_args,save_to_dir = None, target_size = (128,128))
        valGener = validateGenerator(batch_size,str(Path(dataDir,'validate')), 'image', 'label', save_to_dir=None, target_size = (128,128))
        modelUnet = unet(learningRate=learning_rateI,decayRate=decayI, input_size =(windowLen,windowLen,1) )
        model_checkpoint = ModelCheckpoint(str(Path(SaveDir,'unet_CV'+str(cvI)+'.hdf5')), monitor='val_jaccard_loss', mode='min',
                                           verbose=1, save_best_only=True, period=NumEpochEval)
        history1=modelUnet.fit_generator(trainGener,epochs=NumEpochs,steps_per_epoch=int(n_imagesTrain/batch_size),
                                        validation_data=valGener,validation_steps=n_imagesValidate,callbacks=[model_checkpoint])

        with open(str(Path(SaveDir,'history_CV'+str(cvI)+'.pkl')), 'wb') as Results:  # Python 3: open(..., 'wb')
            pickle.dump(
                [history1.history], Results)

        #Loading and testing the model with lowest validation loss
        print('Testing the best U-Net model on testing data and saving the results to: '+str(SaveDir_crops_cv))
        testModel(str(Path(SaveDir, 'unet_CV' + str(cvI) + '.hdf5')), str(Path(dataDir,'test','crops','image')),
                      str(SaveDir_crops_cv))

        #Creating full image mask from the crops predictions
        if cvI < num_CV - 1:
            subjectNums_cvI_testing = subject_nums_shaffled[cvI * int(numSubj / num_CV):cvI * int(numSubj / num_CV) + int(numSubj / num_CV)]
        else:
            subjectNums_cvI_testing = subject_nums_shaffled[cvI * int(numSubj / num_CV):numSubj]

        #Finding the predictions or ICH segmentation for the whole slice
        print('Combining the crops masks to find the full CT mask after performing morphological operations and saving the results to: ' + str(SaveDir_full_cv))
        for subItest in range(0, len(subjectNums_cvI_testing)):
            slicenum_s = hemorrhageDiagnosisArray[hemorrhageDiagnosisArray[:, 0] == subjectNums_cvI_testing[subItest], 1]
            sliceInds = np.where(hemorrhageDiagnosisArray[:, 0] == subjectNums_cvI_testing[
                subItest])  # using the slice index to keep the predictions have the same sequence as the ground truth.
            counterSlice = 0
            for sliceI in range(slicenum_s.size):
                #reading the predicted segmentation for each window
                CTslicePredict = np.zeros((imageLen, imageLen))
                windowOcc = np.zeros((imageLen, imageLen))  # number of predictions for each pixel in the CT scan
                counterCrop = 0
                for i in range(num_Moves):
                    for j in range(num_Moves):
                        windowI = imread(Path(SaveDir_crops_cv,str(subjectNums_cvI_testing[subItest])
                                      + '_' + str(sliceI) +'_'+ str(counterCrop)+ '.png'))
                        windowI = windowI / 255
                        CTslicePredict[int(i * imageLen / (num_Moves + 1)):int(i * imageLen / (num_Moves + 1) + windowLen),
                            int(j * imageLen / (num_Moves + 1)):int(j * imageLen / (num_Moves + 1) + windowLen)]= CTslicePredict[int(i * imageLen / (num_Moves + 1)):int(i * imageLen / (num_Moves + 1) + windowLen),
                            int(j * imageLen / (num_Moves + 1)):int(j * imageLen / (num_Moves + 1) + windowLen)]+windowI
                        windowOcc[int(i * imageLen / (num_Moves + 1)):int(i * imageLen / (num_Moves + 1) + windowLen),
                            int(j * imageLen / (num_Moves + 1)):int(j * imageLen / (num_Moves + 1) + windowLen)]= windowOcc[int(i * imageLen / (num_Moves + 1)):int(i * imageLen / (num_Moves + 1) + windowLen),
                            int(j * imageLen / (num_Moves + 1)):int(j * imageLen / (num_Moves + 1) + windowLen)]+1
                        counterCrop = counterCrop + 1
                CTslicePredict=CTslicePredict / windowOcc *255
                img=np.uint8(CTslicePredict)
                imsave(Path(SaveDir_full_cv,str(subjectNums_cvI_testing[subItest])
                                      + '_' + str(sliceI) + '.png'), img)

                img = np.int16(np.where(img > detectionThreshold, 255, 0))
                img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_closing)  # Filling the gaps
                img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_opening)
                imsave(Path(SaveDir_cv,str(subjectNums_cvI_testing[subItest])
                            + '_' + str(sliceI) + '.png'), np.uint8(img))
                testPredictions[sliceInds[0][counterSlice]] = np.uint8(np.where(img > (0.5*256), 1, 0))
                counterSlice+=1

        K.clear_session()

    CVtestPredictionsAvg = np.where(np.sum(np.sum(testPredictions,axis=1),axis=1) > detectionSen, 1, 0) #

    #Calculating the Final Testing Results for all CV iterations, results for pixel-wise classification
    class_report=np.zeros((numSubj,14))
    for subjI in range(numSubj):
        sliceInds = np.where(hemorrhageDiagnosisArray[:, 0] == subjI)[0]
        class_report[subjI,0]=Jaccard_img(testMasks[sliceInds],testPredictions[sliceInds])
        class_report[subjI,1] = dice_img(testMasks[sliceInds], testPredictions[sliceInds])
        #Results for slice-wise classification
        class_report[subjI,2] = metrics.accuracy_score(testMasksAvg,CVtestPredictionsAvg)
        class_report[subjI,3] = metrics.recall_score(testMasksAvg,CVtestPredictionsAvg, pos_label=1)
        class_report[subjI,4] = metrics.precision_score(testMasksAvg,CVtestPredictionsAvg, pos_label=1)
        class_report[subjI,5] = metrics.f1_score(testMasksAvg,CVtestPredictionsAvg, pos_label=1)
        class_report[subjI,6] = Sens(testMasksAvg,CVtestPredictionsAvg)  # TPR is also known as sensitivity
        class_report[subjI,7] = Speci(testMasksAvg,CVtestPredictionsAvg)  # FPR is one minus the specificity or true negative rate

    class_report[21, :] = np.nan  # this subject has chronic ICH so exclude from results
    print("Final pixel-wise testing: mean Jaccard %.3f (max %.3f, min %.3f, +- %.3f), mean Dice %.3f (max %.3f, min %.3f, +- %.3f)" % (
            np.nanmean(class_report[:,0]), np.nanmax(class_report[:,0]),np.nanmin(class_report[:,0]),np.nanstd(class_report[:,0]),
            np.nanmean(class_report[:,1]),np.nanmax(class_report[:,1]),np.nanmin(class_report[:,1]), np.nanstd(class_report[:,1])))
    print("Final testing: Accuracy %.3f (max %.3f, min %.3f, +- %.3f), Sensi %.4f (max %.3f, min %.3f, +- %.3f), Speci %.4f (max %.3f, min %.3f, +- %.3f))." % (
                np.nanmean(class_report[:,2]),np.nanmax(class_report[:,2]),np.nanmin(class_report[:,2]),np.nanstd(class_report[:,2])
                ,np.nanmean(class_report[:,3]), np.nanmax(class_report[:,3]),np.nanmin(class_report[:,3]),np.nanstd(class_report[:,3])
                ,np.nanmean(class_report[:,3]), np.nanmax(class_report[:,3]), np.nanmin(class_report[:,3]),np.nanstd(class_report[:,3])))

    with open(str(Path(SaveDir,'fullCT_morph' + str(thresholdI),'report.pkl')), 'wb') as Results:  # Python 3: open(..., 'wb')
        pickle.dump(
            [class_report, testMasks,testPredictions], Results)
