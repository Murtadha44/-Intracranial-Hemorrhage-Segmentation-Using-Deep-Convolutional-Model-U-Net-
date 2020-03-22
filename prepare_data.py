import numpy as np, os, random
from pathlib import Path
import pandas as pd
from imageio import imsave
import pickle
import zipfile
import nibabel as nib

def window_ct (ct_scan, w_level=40, w_width=120):
    w_min = w_level - w_width / 2
    w_max = w_level + w_width / 2
    num_slices=ct_scan.shape[2]
    for s in range(num_slices):
        slice_s = ct_scan[:,:,s]
        slice_s = (slice_s - w_min)*(255/(w_max-w_min)) #or slice_s = (slice_s - (w_level-(w_width/2)))*(255/(w_width))
        slice_s[slice_s < 0]=0
        slice_s[slice_s > 255] = 255
        #slice_s=np.rot90(slice_s)
        ct_scan[:,:,s] = slice_s

    return ct_scan

def load_ct_mask(datasetDir,sub_n, window_specs):
    ct_dir_subj = Path(datasetDir, 'ct_scans', "{0:0=3d}.nii".format(sub_n))
    ct_scan_nifti = nib.load(str(ct_dir_subj))
    ct_scan = ct_scan_nifti.get_data()
    ct_scan = window_ct(ct_scan, window_specs[0], window_specs[1])  # Convert the CT scans using a brain window
    # Loading the masks
    masks_dir_subj = Path(datasetDir, 'masks', "{0:0=3d}.nii".format(sub_n))
    masks_nifti = nib.load(str(masks_dir_subj))
    masks = masks_nifti.get_data()
    return ct_scan, masks

def segment_ct(x, imageLen, windowLen, n_moves):
    x_segmented = np.zeros([windowLen, windowLen, n_moves * n_moves], dtype=np.uint8)
    counterCrop = 0
    for i in range(n_moves):
        for j in range(n_moves):
            x_segmented[:, :, counterCrop] = x[int(i * imageLen / (n_moves + 1)):int(
                i * imageLen / (n_moves + 1) + windowLen),
                                            int(j * imageLen / (n_moves + 1)):int(
                                                j * imageLen / (n_moves + 1) + windowLen)]
            counterCrop = counterCrop + 1
    return x_segmented

def prepare_data(dataset_zip_dir, crossvalid_dir, numSubj, imageLen, windowLen, strideLen, n_moves, window_specs):
    #This function is to 1-load the ICH segmentation dataset and unzip it to ich_data.
    #                2-load all CT scans (nifti format) to divide them for training, validation and testing folders
    #                   DataV1\CV0\train
    #                             \validate
    #                             \test
    #                   DataV1\CV1\...
    #                3-Divide each slice into 49 crops using 128x128 window with stride 64.
    #                4-For the training and validation data, saving only the crops that have ICH, whereas for the testing
    #                  data, saving all the crops.
    #


    currentDir=Path(os.getcwd())
    datasetDir=Path(currentDir, 'ich_data', dataset_zip_dir[:-4])
    if os.path.isfile(str(Path(crossvalid_dir,'ICH_DataSegmentV1.pkl')))==False: #means the cross-validation folds was created before
        if os.path.isdir(crossvalid_dir) == False:
            os.mkdir(crossvalid_dir)

        #Load the dataset
        if os.path.exists(dataset_zip_dir)==True:
            if os.path.exists(str(datasetDir))==True:
                print("The dataset is already unzipped!")
            else:
                print("Unzipping dataset, this may take a while!")
                with zipfile.ZipFile(dataset_zip_dir, 'r') as zip_ref:
                    zip_ref.extractall('ich_data')

            #Reading labels
            hemorrhage_diagnosis_df = pd.read_csv(Path(datasetDir, 'hemorrhage_diagnosis_raw_ct.csv'))
            hemorrhage_diagnosis_array = hemorrhage_diagnosis_df._get_values
            '''columns=['PatientNumber','SliceNumber','Intraventricular','Intraparenchymal','Subarachnoid','Epidural',
                                                                                      'Subdural', 'No_Hemorrhage']) '''
            hemorrhage_diagnosis_array[:, 0] = hemorrhage_diagnosis_array[:, 0] - 49  # starting the subject count from 0

            #Training, Validation and testing using 5-fold Crossvalidation
            NumCV=5# number of crossvalidation folds
            subject_nums = np.unique(hemorrhage_diagnosis_array[:, 0])
            subject_nums_shaffled = [31, 39, 21, 33, 34, 5, 54, 2, 67, 68, 53, 29, 44, 76, 59, 73, 77, 71, 61, 69, 50, 32, 6, 37,
                    57, 75, 80, 41, 27, 40, 46, 79, 45, 55, 62, 7, 66, 58, 78, 4, 47, 52, 28, 20, 24, 51, 36, 63, 30,
                    48, 26, 60, 49, 25, 42, 18, 43, 72, 0, 35, 81, 70, 22, 64, 1, 3, 17, 74, 23, 38, 8, 65, 19,
                    56, 9]  # already shuffled #subject# 10 to 16 are missing

            #reading images
            print("Dividing the data for the 5-fold cross-validation:")
            for cvI in range(0,NumCV):
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

                if cvI<NumCV-1:
                    subjectNums_cvI_testing = subject_nums_shaffled[cvI*int(numSubj/NumCV):cvI*int(numSubj/NumCV)+int(numSubj/NumCV)]
                    subjectNums_cvI_trainVal = np.delete(subject_nums_shaffled,range(cvI * int(numSubj / NumCV),cvI * int(numSubj / NumCV) + int(numSubj / NumCV)))
                else:
                    subjectNums_cvI_testing = subject_nums_shaffled[cvI*int(numSubj/NumCV):numSubj]
                    subjectNums_cvI_trainVal = np.delete(subject_nums_shaffled,range(cvI * int(numSubj / NumCV),numSubj))

                counterI=0

                #Training CT scans
                for subItrain in range(int(numSubj / NumCV),len(subjectNums_cvI_trainVal)):  # take only the 3 folds for training
                    sliceNums=hemorrhage_diagnosis_array[hemorrhage_diagnosis_array[:,0]==subjectNums_cvI_trainVal[subItrain],1]

                    # Loading the CT scans and the masks
                    ct_scan, masks= load_ct_mask(datasetDir, subjectNums_cvI_trainVal[subItrain]+49, window_specs)

                    NoHemorrhage=hemorrhage_diagnosis_array[hemorrhage_diagnosis_array[:, 0] == subjectNums_cvI_trainVal[subItrain], 7]
                    SlicesWithoutHemm = np.nonzero(NoHemorrhage)
                    randomSlice=random.choice(SlicesWithoutHemm[0])
                    for sliceI in range(0,sliceNums.size):
                        if NoHemorrhage[sliceI] == 0 or sliceI==randomSlice: # Saving only the windows that have hemorrhage or
                                                                             # if the slice is selected randomly then save its crops
                            #segmenting the ct scan and the masks
                            x_ct_segmented = segment_ct(ct_scan[:, :, sliceI], imageLen, windowLen, n_moves)
                            x_mask_segment = segment_ct(masks[:, :, sliceI], imageLen, windowLen, n_moves)

                            #Saving only the windows that have hemorrhage
                            for i in range(n_moves*n_moves):
                                if sum(sum(x_mask_segment[:, :, i]))>0:
                                    imsave(Path(crossvalid_dir,'CV' +str(cvI),'train','image',str(counterI) + '.png'), np.uint8(x_ct_segmented[:,:,i]))
                                    imsave(Path(crossvalid_dir,'CV' +str(cvI),'train','label',str(counterI) + '.png'), np.uint8(x_mask_segment[:, :,i]))
                                    counterI = counterI + 1


                # Validation CT scans
                counterI = 0
                for subIvalidate in range(0, int(numSubj / NumCV)):  # take only the first fold for validation
                    sliceNums = hemorrhage_diagnosis_array[hemorrhage_diagnosis_array[:, 0] == subjectNums_cvI_trainVal[subIvalidate], 1]
                    
                    # Loading the CT scans and the masks
                    ct_scan, masks= load_ct_mask(datasetDir, subjectNums_cvI_trainVal[subIvalidate]+49, window_specs)

                    NoHemorrhage=hemorrhage_diagnosis_array[hemorrhage_diagnosis_array[:, 0] == subjectNums_cvI_trainVal[subIvalidate], 7]
                    SlicesWithoutHemm = np.nonzero(NoHemorrhage)
                    for sliceI in range(0,sliceNums.size):
                        if NoHemorrhage[sliceI] == 0: # Saving only the windows that have hemorrhage
                            #segmenting the ct scan and the masks
                            x_ct_segmented = segment_ct(ct_scan[:, :, sliceI], imageLen, windowLen, n_moves)
                            x_mask_segment = segment_ct(masks[:, :, sliceI], imageLen, windowLen, n_moves)

                            #Saving only the windows that have hemorrhage
                            for i in range(n_moves*n_moves):
                                if sum(sum(x_mask_segment[:, :, i]))>0:
                                    imsave(Path(crossvalid_dir,'CV' +str(cvI),'validate','image',str(counterI) + '.png'), np.uint8(x_ct_segmented[:,:,i]))
                                    imsave(Path(crossvalid_dir,'CV' +str(cvI),'validate','label',str(counterI) + '.png'), np.uint8(x_mask_segment[:, :,i]))
                                    counterI = counterI + 1

                # Testing CT scans
                for subItest in range(0, len(subjectNums_cvI_testing)):
                    sliceNums = hemorrhage_diagnosis_array[
                        hemorrhage_diagnosis_array[:, 0] == subjectNums_cvI_testing[subItest], 1]

                    # Loading the CT scans and the masks
                    ct_scan, masks = load_ct_mask(datasetDir, subjectNums_cvI_testing[subItest] + 49, window_specs)
                    counterI = 0
                    for sliceI in range(0, sliceNums.size):
                        # segmenting the ct scan and the masks
                        x_ct_segmented = segment_ct(ct_scan[:, :, sliceI], imageLen, windowLen, n_moves)
                        x_mask_segment = segment_ct(masks[:, :, sliceI], imageLen, windowLen, n_moves)

                        imsave(Path(crossvalid_dir,'CV' + str(cvI),'test','fullCT','image', str(subjectNums_cvI_testing[subItest])
                                          + '_' + str(counterI) + '.png'),np.uint8(ct_scan[:, :, sliceI]))
                        imsave(Path(crossvalid_dir,'CV' + str(cvI),'test','fullCT','label', str(subjectNums_cvI_testing[subItest])
                                          + '_' + str(counterI) + '.png'),np.uint8(masks[:, :, sliceI]))

                        counterCrop = 0
                        for i in range(n_moves * n_moves):
                            imsave(Path(crossvalid_dir, 'CV' + str(cvI), 'test','crops', 'image',
                                        str(subjectNums_cvI_testing[subItest])+ '_' + str(counterI) + '_' +
                                        str(counterCrop) + '.png'), np.uint8(x_ct_segmented[:, :, i]))
                            imsave(Path(crossvalid_dir, 'CV' + str(cvI), 'test','crops', 'label',
                                        str(subjectNums_cvI_testing[subItest])+ '_' + str(counterI) + '_' +
                                        str(counterCrop) + '.png'), np.uint8(x_mask_segment[:, :, i]))
                            counterCrop = counterCrop + 1

                        counterI = counterI + 1

            ############################################Saving the CT slices to ndarrays#####################################

            AllCTscans = np.zeros([hemorrhage_diagnosis_array.shape[0], imageLen, imageLen], dtype=np.uint8)
            Allsegment = np.zeros([hemorrhage_diagnosis_array.shape[0], imageLen, imageLen], dtype=np.uint8)
            counterI=0

            for s_num in subject_nums:
                sliceNums=hemorrhage_diagnosis_array[hemorrhage_diagnosis_array[:,0]==s_num,1]

                # Loading the CT scans and the masks
                ct_scan, masks = load_ct_mask(datasetDir, s_num + 49, window_specs)

                for sliceI in range(0,sliceNums.size):
                    AllCTscans[counterI] = ct_scan[:,:,sliceI]
                    Allsegment[counterI] = masks[:,:,sliceI]
                    counterI=counterI+1

            with open(str(Path(crossvalid_dir,'ICH_DataSegmentV1.pkl')), 'wb') as Dataset1:  # Python 3: open(..., 'wb')
                    pickle.dump(
                        [hemorrhage_diagnosis_array, AllCTscans, Allsegment, subject_nums_shaffled], Dataset1)

        else:
            print("Zipped dataset is not in the current directory. "
                  "Download the dataset from https://physionet.org/content/ct-ich/1.3.1/ then move it to same directory with main.py")
    else:
        print("The dataset is already downloaded and the cross-validation folds were created!")
