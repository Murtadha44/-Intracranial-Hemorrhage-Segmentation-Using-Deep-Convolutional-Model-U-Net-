import numpy as np, os, random
from pathlib import Path
import pandas as pd
from scipy.misc import imread, imsave
import pickle
import urllib.request
import zipfile

def prepare_data(crossvalid_dir='DataV1'):
    #This function is to 1-download the ICH segmentation dataset and unzip it to ich_data.
    #                2-load all CT scans to divide them for training, validation and testing folders
    #                   DataV1\CV0\train
    #                             \validate
    #                             \test
    #                   DataV1\CV1\...
    #                3-Divide each slice into 49 crops using 160x160 window with stride 80.
    #


    currentDir=Path(os.getcwd())
    datasetDir=str(Path(currentDir,'ich_data',
                        'computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.0.0'))
    if os.path.isfile(str(Path(crossvalid_dir,'ICH_DataSegmentV1.pkl')))==False: #means the cross-validation folds was created before
        if os.path.isdir(str(Path(crossvalid_dir))) == False:
            os.mkdir(str(Path(crossvalid_dir)))

        #Download the dataset
        url = 'https://physionet.org/static/published-projects/ct-ich/computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.0.0.zip'
        if os.path.isdir(str(Path('ich_data'))) == False:
            if os.path.exists(str(Path('ich_data.zip')))==False:
                print("Loading the ICH segmentation dataset from physionet.org")
                urllib.request.urlretrieve(url, "ich_data.zip")
            else:
                print("Unzipping dataset, this may take a while!")
                with zipfile.ZipFile('ich_data.zip', 'r') as zip_ref:
                    zip_ref.extractall('ich_data')

        numSubj=82
        imageLen = 640
        windowLen=160
        strideLen=80
        noMoves = int(imageLen/strideLen)-1

        #Training, Validation and testing using 5-fold Crossvalidation
        NoCV=5# number of crossvalidation folds
        sNos=[31, 39, 21, 33, 34, 5,54,2,67,15,68,10,53,29,44,76,59,73,77,71,61,69,50,32,6,37,57,75,
              80,41,27,16,40,46,79,13,45,55,62,7,66,58,78,4,47,52,28,20,24,51,36,63,30,48,26,60,49,
              25,42,18,43,14,72,0,35,81,70,22,64,1,3,17,74,23,38,12,8,65,19,56,9,11] #already shuffled


        #Reading labels
        hemorrhage_diagnosis_df = pd.read_csv(Path(datasetDir, 'hemorrhage_diagnosis.csv'))
        hemorrhage_diagnosis_array = hemorrhage_diagnosis_df._get_values
        '''columns=['PatientNumber','SliceNumber','Intraventricular','Intraparenchymal','Subarachnoid','Epidural',
                                                                                  'Subdural', 'No_Hemorrhage']) '''
        hemorrhage_diagnosis_array[:, 0] = hemorrhage_diagnosis_array[:, 0] - 49  # starting the subject count from 0

        #reading images
        print("Dividing the data for the 5-fold cross-validation:")
        for cvI in range(0,NoCV):
            print("Working on fold #"+str(cvI))
            print("The full CT slices and the crops will be save to:" + str(Path(crossvalid_dir, 'CV' + str(cvI))))
            if os.path.isdir(str(Path(crossvalid_dir,'CV'+str(cvI)))) == False:
                os.mkdir(str(Path(crossvalid_dir,'CV'+str(cvI))))
                os.mkdir(str(Path(crossvalid_dir,'CV'+str(cvI),'train')))
                os.mkdir(str(Path(crossvalid_dir, 'CV' + str(cvI), 'train', 'image')))
                os.mkdir(str(Path(crossvalid_dir, 'CV' + str(cvI), 'train','label')))
                os.mkdir(str(Path(crossvalid_dir,'CV' + str(cvI), 'test')))
                os.mkdir(str(Path(crossvalid_dir,'CV' + str(cvI), 'test','fullCT')))
                os.mkdir(str(Path(crossvalid_dir, 'CV' + str(cvI), 'test', 'crops')))
                os.mkdir(str(Path(crossvalid_dir, 'CV' + str(cvI), 'test','fullCT', 'image')))
                os.mkdir(str(Path(crossvalid_dir, 'CV' + str(cvI), 'test','fullCT','label')))
                os.mkdir(str(Path(crossvalid_dir, 'CV' + str(cvI), 'test', 'crops', 'image')))
                os.mkdir(str(Path(crossvalid_dir, 'CV' + str(cvI), 'test', 'crops','label')))
                os.mkdir(str(Path(crossvalid_dir,'CV' + str(cvI), 'validate')))
                os.mkdir(str(Path(crossvalid_dir, 'CV' + str(cvI), 'validate', 'image')))
                os.mkdir(str(Path(crossvalid_dir, 'CV' + str(cvI), 'validate','label')))

            if cvI<NoCV-1:
                subjectNos_cvI_testing=sNos[cvI*int(numSubj/NoCV):cvI*int(numSubj/NoCV)+int(numSubj/NoCV)]
                subjectNos_cvI_trainVal = np.delete(sNos,range(cvI * int(numSubj / NoCV),cvI * int(numSubj / NoCV) + int(numSubj / NoCV)))
            else:
                subjectNos_cvI_testing = sNos[cvI*int(numSubj/NoCV):numSubj]
                subjectNos_cvI_trainVal = np.delete(sNos,range(cvI * int(numSubj / NoCV),numSubj))

            counterI=0
            #Training CT scans
            for subItrain in range(int(numSubj / NoCV),len(subjectNos_cvI_trainVal)):  # take only the 3 folds for training
                sliceNos=hemorrhage_diagnosis_array[hemorrhage_diagnosis_array[:,0]==subjectNos_cvI_trainVal[subItrain],1]
                datasetDirSubj = Path(datasetDir,
                        'Patients_CT', "{0:0=3d}".format(subjectNos_cvI_trainVal[subItrain]+49))
                NoHemorrhage=hemorrhage_diagnosis_array[hemorrhage_diagnosis_array[:, 0] == subjectNos_cvI_trainVal[subItrain], 7]
                SlicesWithoutHemm = np.nonzero(NoHemorrhage)
                randomSlice=random.choice(SlicesWithoutHemm[0])
                for sliceI in range(0,sliceNos.size):
                    x_original=np.zeros([windowLen,windowLen,noMoves*noMoves],dtype=np.uint8)
                    x_segment = np.zeros([windowLen, windowLen, noMoves*noMoves],dtype=np.uint8)
                    if NoHemorrhage[sliceI] == 0: #Saving only the windows that have hemorrhage
                        img_path=Path(datasetDirSubj,'brain',str(sliceNos[sliceI]) + '.jpg')
                        img = imread(img_path)
                        ##img = imresize(img,new_size)
                        x = img[5:-5, 5:-5]  # clipping the image to size 640 (new_size)
                        counterCrop=0
                        for i in range(noMoves):
                            for j in range(noMoves):
                                x_original[:, :, counterCrop] = x[int(i*imageLen/(noMoves+1)):int(i*imageLen/(noMoves+1)+windowLen),
                                                                int(j*imageLen/(noMoves+1)):int(j*imageLen/(noMoves+1)+windowLen)]
                                counterCrop=counterCrop+1

                        #Reading the segmentation for a given slice
                        segment_path = Path(datasetDirSubj,'brain',str(sliceNos[sliceI])+ '_HGE_Seg.jpg')
                        if os.path.exists(str(segment_path)):
                            img = imread(segment_path)
                            ##img = imresize(img,new_size)
                            x = np.where(img > 128, 255,
                                         0)  # Because of the resize the image has some values that are not 0 or 255, so make them 0 or 255
                            x = x[5:-5, 5:-5]  # clipping the image to size 640 (new_size)
                            counterCrop = 0
                            for i in range(noMoves):
                                for j in range(noMoves):
                                    x_segment[:, :, counterCrop] =  x[int(i*imageLen/(noMoves+1)):int(i*imageLen/(noMoves+1)+windowLen),
                                                                int(j*imageLen/(noMoves+1)):int(j*imageLen/(noMoves+1)+windowLen)]
                                    counterCrop = counterCrop + 1

                            #Saving only the windows that have hemorrhage
                            for i in range(noMoves*noMoves):
                                if sum(sum(x_segment[:, :, i]))>0:
                                    x_original[-1, -1, i] = 255  # having a pixel of 255 so the array will not rescaled to 0-255 bym imsave
                                    imsave(Path(crossvalid_dir,'CV' +str(cvI),'train','image',str(counterI) + '.png'), x_original[:,:,i])
                                    imsave(Path(crossvalid_dir,'CV' +str(cvI),'train','label',str(counterI) + '.png'), x_segment[:, :,i])
                                    counterI = counterI + 1
                        else:
                            print("Error: Segmentation image was not found.")
                    elif sliceI==randomSlice: #if the slice is selected randomly then save its crops
                        img_path=Path(datasetDirSubj,'brain',str(sliceNos[sliceI]) + '.jpg')
                        img = imread(img_path)
                        ##img = imresize(img,new_size)
                        x = img[5:-5, 5:-5]  # clipping the image to size 640 (new_size)
                        counterCrop=0
                        for i in range(noMoves):
                            for j in range(noMoves):
                                x_original[:, :, counterCrop] = x[int(i*imageLen/(noMoves+1)):int(i*imageLen/(noMoves+1)+windowLen),
                                                                int(j*imageLen/(noMoves+1)):int(j*imageLen/(noMoves+1)+windowLen)]
                                counterCrop=counterCrop+1

                        #Reading the segmentation for a given slice
                        x = np.zeros((640, 640),dtype=np.uint8) # No hemorrhage for detected in this slice so all zeros
                        counterCrop = 0
                        for i in range(noMoves):
                            for j in range(noMoves):
                                x_segment[:, :, counterCrop] =  x[int(i*imageLen/(noMoves+1)):int(i*imageLen/(noMoves+1)+windowLen),
                                                            int(j*imageLen/(noMoves+1)):int(j*imageLen/(noMoves+1)+windowLen)]
                                counterCrop = counterCrop + 1

                        #Saving only the windows that have hemorrhage
                        for i in range(noMoves*noMoves):
                            x_original[-1, -1, i] = 255  # having a pixel of 255 so the array will not rescaled to 0-255 bym imsave
                            imsave(Path(crossvalid_dir,'CV' +str(cvI),'train','image',str(counterI) + '.png'), x_original[:,:,i])
                            imsave(Path(crossvalid_dir,'CV' +str(cvI),'train','label',str(counterI) + '.png'), x_segment[:, :,i])
                            counterI = counterI + 1

            # Validation CT scans
            counterI = 0
            for subIvalidate in range(0, int(numSubj / NoCV)):  # take only the first fold for validation
                sliceNos = hemorrhage_diagnosis_array[hemorrhage_diagnosis_array[:, 0] == subjectNos_cvI_trainVal[subIvalidate], 1]
                datasetDirSubj = Path(datasetDir,
                        'Patients_CT', "{0:0=3d}".format(subjectNos_cvI_trainVal[subIvalidate]+49))
                NoHemorrhage = hemorrhage_diagnosis_array[hemorrhage_diagnosis_array[:, 0] == subjectNos_cvI_trainVal[subIvalidate], 7]

                for sliceI in range(0,sliceNos.size):
                    x_original=np.zeros([windowLen,windowLen,noMoves*noMoves],dtype=np.uint8)
                    x_segment = np.zeros([windowLen, windowLen, noMoves*noMoves],dtype=np.uint8)
                    if NoHemorrhage[sliceI] == 0:  # Saving only the windows that have hemorrhage. Thus it will be used to select the model with highest detection
                        img_path = Path(datasetDirSubj,'brain', str(sliceNos[sliceI]) + '.jpg')
                        img = imread(img_path)
                        ##img = imresize(img,new_size)
                        x = img[5:-5, 5:-5]  # clipping the image to size 640 (new_size)
                        counterCrop = 0
                        for i in range(noMoves):
                            for j in range(noMoves):
                                x_original[:, :, counterCrop] = x[int(i * imageLen / (noMoves + 1)):int(
                                    i * imageLen / (noMoves + 1) + windowLen),
                                                                int(j * imageLen / (noMoves + 1)):int(
                                                                    j * imageLen / (noMoves + 1) + windowLen)]
                                counterCrop = counterCrop + 1

                        # Reading the segmentation for a given slice
                        segment_path = Path(datasetDirSubj,'brain', str(sliceNos[sliceI]) + '_HGE_Seg.jpg')
                        if os.path.exists(str(segment_path)):
                            img = imread(segment_path)
                            ##img = imresize(img,new_size)
                            x = np.where(img > 128, 255,
                                         0)  # Because of the resize the image has some values that are not 0 or 255, so make them 0 or 255
                            x = x[5:-5, 5:-5]  # clipping the image to size 640 (new_size)
                            counterCrop = 0
                            for i in range(noMoves):
                                for j in range(noMoves):
                                    x_segment[:, :, counterCrop] = x[int(i * imageLen / (noMoves + 1)):int(
                                        i * imageLen / (noMoves + 1) + windowLen),
                                                                   int(j * imageLen / (noMoves + 1)):int(
                                                                       j * imageLen / (noMoves + 1) + windowLen)]
                                    counterCrop = counterCrop + 1

                            # Saving only the windows that have hemorrhage
                            for i in range(noMoves * noMoves):
                                if sum(sum(x_segment[:, :, i])) > 0:
                                    x_original[-1, -1, i] = 255  # having a pixel of 255 so the array will not rescaled to 0-255 bym imsave
                                    imsave(Path(crossvalid_dir,'CV' +str(cvI),'validate','image',str(counterI) + '.png'),
                                                      x_original[:, :, i])
                                    imsave(Path(crossvalid_dir,'CV' +str(cvI),'validate','label',str(counterI) + '.png'),
                                                      x_segment[:, :, i])
                                    counterI = counterI + 1
                        else:
                            print("Error: Segmentation image was not found.")

            # Testing CT scans
            for subItest in range(0, len(subjectNos_cvI_testing)):
                sliceNos = hemorrhage_diagnosis_array[hemorrhage_diagnosis_array[:, 0] == subjectNos_cvI_testing[subItest], 1]
                datasetDirSubj = Path(datasetDir,
                        'Patients_CT', "{0:0=3d}".format(subjectNos_cvI_testing[subItest]+49))

                counterI = 0
                for sliceI in range(0, sliceNos.size):
                    x_original = np.zeros([windowLen, windowLen, noMoves * noMoves],dtype=np.uint8)
                    x_segment = np.zeros([windowLen, windowLen, noMoves * noMoves],dtype=np.uint8)

                    img_path = Path(datasetDirSubj,'brain', str(sliceNos[sliceI]) + '.jpg')
                    img = imread(img_path)
                    ##img = imresize(img,new_size)
                    x = img[5:-5, 5:-5]  # clipping the image to size 640 (new_size)
                    imsave(Path(crossvalid_dir,'CV' + str(cvI),'test','fullCT','image', str(subjectNos_cvI_testing[subItest])
                                      + '_' + str(counterI) + '.png'),x)
                    counterCrop = 0
                    for i in range(noMoves):
                        for j in range(noMoves):
                            x_original[:, :, counterCrop] = x[int(i*imageLen/(noMoves+1)):int(i*imageLen/(noMoves+1)+windowLen),
                                                                int(j*imageLen/(noMoves+1)):int(j*imageLen/(noMoves+1)+windowLen)]
                            counterCrop = counterCrop + 1

                    # Reading the segmentation for a given slice
                    segment_path = Path(datasetDirSubj,'brain', str(sliceNos[sliceI]) + '_HGE_Seg.jpg')
                    if os.path.exists(str(segment_path)):
                        img = imread(segment_path)
                        ##img = imresize(img,new_size)
                        x = np.where(img > 128, 255,
                                     0)  # Because of the resize the image has some values that are not 0 or 255, so make them 0 or 255
                        x = x[5:-5, 5:-5]  # clipping the image to size 640 (new_size)
                        imsave(Path(crossvalid_dir,'CV' + str(cvI),'test','fullCT','label', str(subjectNos_cvI_testing[subItest])
                                      + '_' + str(counterI) + '.png'), x)
                        counterCrop = 0
                        for i in range(noMoves):
                            for j in range(noMoves):
                                x_segment[:, :, counterCrop] = x[int(i*imageLen/(noMoves+1)):int(i*imageLen/(noMoves+1)+windowLen),
                                                                int(j*imageLen/(noMoves+1)):int(j*imageLen/(noMoves+1)+windowLen)]
                                x_original[-1, -1, counterCrop] = 255  # having a pixel of 255 so the array will not rescaled to 0-255 bym imsave
                                imsave(Path(crossvalid_dir,'CV' + str(cvI),'test','crops','image',  str(subjectNos_cvI_testing[subItest])
                                      + '_' + str(counterI) +'_'+ str(counterCrop)+ '.png'),
                                                  x_original[:, :, counterCrop])
                                imsave(Path(crossvalid_dir,'CV' + str(cvI),'test','crops','label',  str(subjectNos_cvI_testing[subItest])
                                      + '_' + str(counterI) +'_'+ str(counterCrop)+  '.png'),
                                                  x_segment[:, :, counterCrop])
                                counterCrop = counterCrop + 1
                    else:  # no hemorrhage then save black images
                        imsave(Path(crossvalid_dir,'CV' + str(cvI),'test','fullCT','label',str(subjectNos_cvI_testing[subItest])
                                      + '_' + str(counterI) + '.png'), np.zeros([imageLen, imageLen],dtype=np.uint8))
                        counterCrop = 0
                        for i in range(noMoves * noMoves):
                            x_original[-1, -1,counterCrop] = 255 #having a pixel of 255 so the array will not rescaled to 0-255 bym imsave
                            imsave(Path(crossvalid_dir,'CV' + str(cvI),'test','crops','image', str(subjectNos_cvI_testing[subItest])
                                      + '_' + str(counterI) +'_'+ str(counterCrop)+ '.png'),
                                              x_original[:, :, counterCrop])
                            imsave(Path(crossvalid_dir,'CV' + str(cvI),'test','crops','label', str(subjectNos_cvI_testing[subItest])
                                      + '_' + str(counterI) +'_'+ str(counterCrop)+  '.png'),
                                              np.zeros([windowLen, windowLen],dtype=np.uint8))
                            counterCrop = counterCrop + 1
                    counterI = counterI + 1

        ############################################Saving the CT slices to ndarrays#####################################

        AllCTscans = np.zeros([hemorrhage_diagnosis_array.shape[0], imageLen, imageLen], dtype=np.uint8)
        Allsegment = np.zeros([hemorrhage_diagnosis_array.shape[0], imageLen, imageLen], dtype=np.uint8)
        counterI=0
        for sNo in range(0,numSubj):
            sliceNos=hemorrhage_diagnosis_array[hemorrhage_diagnosis_array[:,0]==sNo,1]
            datasetDirSubj = Path(datasetDir,
                        'Patients_CT',"{0:0=3d}".format(sNo+49))
            for sliceI in range(0,sliceNos.size):
                img_path=Path(datasetDirSubj,'brain',str(sliceNos[sliceI]) + '.jpg')
                img = imread(img_path)
                ##img = imresize(img,new_size)
                x = img[5:-5, 5:-5]  # clipping the image to size 640 (new_size)
                AllCTscans[counterI] = x

                #Saving the segmentation for a given slice
                segment_path = Path(datasetDirSubj,'brain',str(sliceNos[sliceI]) + '_HGE_Seg.jpg')
                if os.path.exists(str(segment_path)):
                    img = imread(segment_path)
                    ##img = imresize(img,new_size)
                    x = np.where(img > 128, 255,
                                 0)  # Because of the resize the image has some values that are not 0 or 255, so make them 0 or 255
                    x = x[5:-5, 5:-5]  # clipping the image to size 640 (new_size)
                    Allsegment[counterI] = x
                else:
                    x=np.zeros([imageLen,imageLen],dtype=np.uint8)
                    Allsegment[counterI] = x

                counterI=counterI+1

        with open(str(Path(crossvalid_dir,'ICH_DataSegmentV1.pkl')), 'wb') as Dataset1:  # Python 3: open(..., 'wb')
                pickle.dump(
                    [hemorrhage_diagnosis_array, AllCTscans, Allsegment, sNos], Dataset1)
    else:
        print("The dataset is already downloaded and the cross-validation folds were created!")
