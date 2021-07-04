clear all;

addpath(genpath('liblinear-2.1'));
addpath('general_function\');
addpath('libraryPubliced\tool\liblinear\');
addpath('libraryPubliced\tool\liblinear\matlab\');
%addpath('libraryPubliced\tool\libsvm\matlab\');


run('I:\wangyancheng\code\vlfeat-0.9.21\toolbox\vl_setup');

clusterfun = 'no'; % or 'dplim', or 'no', or 'kmlim';
numClass = 60;
numCluster = 30;
numGmm = 4;
numBlock = 1;
gmmSamRatio = 0.6;

isPCA = 0;
isFVNorm = 0;
isFV = 2;
view_trn = [];
view_tst = [];
sub_trn = [];
sub_tst = [];
path = 'G:\wangyancheng\ntu_rgbd\dynamic\feature_ntu_view_share_MBB\';% data path
index_s = ['01';'02';'03';'04';'05';'06';'07';'08';'09';'10';'11';'12';'13';'14';'15';'16';'17'];
index_c = ['01';'02';'03'];
index_p = ['01';'02';'04';'05';'08';'09';'13';'14';'15';'16';'17';'18';'19';'25';'27';'28';'31';'34';'35';'38';...
    '03';'06';'07';'10';'11';'12';'20';'21';'22';'23';'24';'26';'29';'30';'32';'33';'36';'37';'39';'40'];%
index_r = ['01';'02'];
index_a = ['01';'02';'03';'04';'05';'06';'07';'08';'09';'10';'11';'12';'13';'14';'15';'16';'17';'18';'19';'20';...
    '21';'22';'23';'24';'25';'26';'27';'28';'29';'30';'31';'32';'33';'34';'35';'36';'37';'38';'39';'40';...
    '41';'42';'43';'44';'45';'46';'47';'48';'49';'50';'51';'52';'53';'54';'55';'56';'57';'58';'59';'60'];
%data_train = zeros(4096,5,11,length(dir([path '*_v01*']))+length(dir([path '*_v02*'])));
data_train = [];
data_test = [];
idx_trn =1;
idx_tst = 1;
lab_trn = [];
lab_tst = [];
dimension = 4096;
dPCA = 4096;
sizeBlock = dimension/numBlock;
%--------- extract the data
for i = 1:length(index_s)
    i
    for j = 1:length(index_c)
        for k = 1:length(index_p)
            for m =1:length(index_r)
                for d = 1:length(index_a)
                    filename = ['S0',index_s(i,:), 'C0' ,index_c(j,:),'P0',index_p(k,:),'R0',index_r(m,:),'A0',index_a(d,:),'.mat'];
                    if exist([path filename],'file')
                        if  j>1
                            dt_train1 = load([path filename]);
                            data = reshape(cell2mat(dt_train1.data),[4096,5,11,2]);
                            data_train(:,:,:,idx_trn) = squeeze(data(:,:,:,1));
                            idx_trn = idx_trn+1;
                            lab_trn = [lab_trn;d];
                            view_trn = [view_trn;j];
                            sub_trn = [sub_trn;j];
                        else
                            dt_test = load([path filename]);
                            data = reshape(cell2mat(dt_test.data),[4096,5,11,2]);
                            data_test(:,:,:,idx_tst) = squeeze(data(:,:,:,1));
                            lab_tst = [lab_tst;d];
                            idx_tst = idx_tst+1;
                            view_tst = [view_tst;j];
                            sub_tst = [sub_tst;j];
                        end
                    end
                end
            end
        end
    end
end

%------- PCA --------%
if isPCA
    fprintf('----PCA----\n');
    dt = reshape(data_train,[4096 5*11*size(data_train,4)]);
    [pc,score,latent,tsquare] = pca(dt');
    after_pca = score(:,1:dPCA);
    dt = reshape(data_test,[4096 5*11*size(data_test,4)]);
    test_pca = dt' * pc(:,1:dPCA);
    sample.train = reshape(after_pca',[dPCA 5 11 numSam_train]);
    sample.test = reshape(test_pca',[dPCA 5 11 numSam_test]);
else
    sample.train = data_train;
    sample.test = data_test;
    dPCA = dimension;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
numSam_train = size(data_train,4);
numSam_test = size(data_test,4);
numSam = numSam_train + numSam_test;
sample.all = cat(4,sample.train,sample.test);  % help cat  [4096 5 11 N]

clear data_train;
clear data_test;
            
            col = 11;row = 1;
%------ cluster patches of training sample in iLayer ------%
%samCluster_all = [reshape(sample.train,dPCA,55*numSam_train)]';

Lab123 = repmat(lab_trn,1,11)';
samImgLabel = Lab123(:);  %每个特征对应的标签
for iSeg = 1:1 %==============
    samCluster = [reshape(sample.train(:,iSeg,:,:),dPCA,11*numSam_train)]'; %[N1*11 4096]
    
    if ~strcmp(clusterfun,'no')
        if strcmp(clusterfun,'dplim')
            res_cluster = func_dpCluster(samCluster,numCluster);
        end
        if strcmp(clusterfun,'kmlim')
            res_cluster = func_kmCluster(samCluster,numCluster,10000,1);
        end
        
        for i = 1:numCluster
            for j = 1:numClass
                numInCluster(i) = length(find(res_cluster.labels == i));  %各个簇所含特征数量
                idxClassJ = find(samImgLabel == j);                       %标签j在所有特征中的位置
                clusterLabelForClassJ = res_cluster.labels(idxClassJ);     %标签j下的特征所属簇
                numEachClInClass(i,j) = length(find(clusterLabelForClassJ == i)); %第i个簇中第j标签的特征数量
            end
            ratioEachCl(i,:) = numEachClInClass(i,:) / numInCluster(i);
            stdEachCl(i) = std(ratioEachCl(i,:));
        end
        [stdSort,idxClusterLabelSort] = sort(stdEachCl);
        
        
        
        %%%%%%%%%%%%%% used to select sample for GMM
        
        %------ select patches ------%
        %for iSel = 1:numClSelect
        idx_selete = find(res_cluster.labels == idxClusterLabelSort(1));
        samGmm = samCluster;
        samGmm(idx_selete,:) = [];
        %end
        clear samCluster;
        
        %%%%%%%%%%%%%%% used to select sample for FV
        idxCenterChoose = idxClusterLabelSort(1);
        for iSam = 1:numSam
            fvSam = [squeeze(sample.all(:,iSeg,:,iSam))]';
            samDist = func_distCenter(fvSam,res_cluster.centersFea);
            for i = 1:row*col
                samLabel(i) = find(samDist(i,:) == min(samDist(i,:)), 1 );
            end
            idxSelect = find(samLabel == idxCenterChoose);
            if length(idxSelect) > row*col*(1-gmmSamRatio)
                iidx = randperm(length(idxSelect));
                nnumSel = row*col*(1-gmmSamRatio);
                idxSelect = idxSelect(iidx(1:nnumSel));
            end
            fvSam(idxSelect,:) = [];
            %-- store the selected samples for fv---%
            if ~isdir('sampleforFV\')
                mkdir('sampleforFV\');
            end
            fvSamName = ['FV_Sample' num2str(iSam)];
            save(['sampleforFV\',fvSamName,'.mat'],'fvSam');
            disp(['Got ',num2str(iSam),'/',num2str(numSam),' samples' 'sample data for FV .....']);
        end
        
        disp('finished selecting all samples for GMM & FV');
        
        %--对选取的特征进行分块处理--%
        for iBlock = 1:numBlock
            sizeBlock = dPCA/numBlock;
            samgmm = samGmm';
            dsamgmm = samgmm((iBlock-1)*sizeBlock+1:iBlock*sizeBlock,:);
            %------ learning gmm ------%
            disp('Computing the gmm model......');
            [gmmComp.mean,gmmComp.covariances,gmmComp.priors] = vl_gmm(dsamgmm,numGmm);
            if ~isdir('gmmComp\')
                mkdir('gmmComp\');
            end
            save(['gmmComp\', 'resGmm_' num2str(iSeg) '_iBlock' num2str(iBlock) '.mat'],'gmmComp');
            disp('gmmComp have been saved!');
            
            %------ fisher vector coding ------%
            for iSam = 1:numSam
                fvSamName = ['FV_Sample' num2str(iSam)];
                load(['sampleforFV\',fvSamName,'.mat'],'fvSam');
                sub_fvSam = fvSam(:,(iBlock-1)*sizeBlock+1:iBlock*sizeBlock);
                if isFVNorm
                    feaFV = vl_fisher(sub_fvSam',gmmComp.mean,gmmComp.covariances,gmmComp.priors,'Normalized');
                else
                    feaFV = vl_fisher(sub_fvSam',gmmComp.mean,gmmComp.covariances,gmmComp.priors,'Improved');
                end
                %-- store the fisher vector---%
                if ~isdir('finalFea\')
                    mkdir('finalFea\');
                end
                if iSam <= numSam_train
                    feaName = ['FV_feaTr' num2str(iSam) '_seg' num2str(iSeg) '_iBlock' num2str(iBlock)];
                else
                    feaName = ['FV_feaTs' num2str(iSam-numSam_train) '_seg' num2str(iSeg) '_iBlock' num2str(iBlock)];
                end
                save(['finalFea\',feaName,'.mat'],'feaFV');
                
            end
        end
        
    else
        samGmm = samCluster;%[N1*11 4096]
        clear samCluster;
        %------ learning gmm ------%
        for iBlock = 1:numBlock
            samgmm = samGmm';%[4096 N1*11]
            dsamgmm = samgmm((iBlock-1)*sizeBlock+1:iBlock*sizeBlock,:);
            disp('Computing the gmm model......');
            [gmmComp.mean,gmmComp.covariances,gmmComp.priors] = vl_gmm(dsamgmm,numGmm);
            if ~isdir('gmmComp\')
                mkdir('gmmComp\');
            end
            save(['gmmComp\', 'resGmm_' num2str(iSeg) '_iBlock' num2str(iBlock) '.mat'],'gmmComp');
            disp('gmmComp have been saved!');           

            %----- FV coding -----%
            for iSam = 1:numSam
                dSam = [squeeze(sample.all((iBlock-1)*sizeBlock+1:iBlock*sizeBlock,iSeg,:,iSam))]';
                
                if isFVNorm
                    feaFV = vl_fisher(dSam',gmmComp.mean,gmmComp.covariances,gmmComp.priors,'Normalized');
                else
                    feaFV = vl_fisher(dSam',gmmComp.mean,gmmComp.covariances,gmmComp.priors,'Improved');
                end
                %-- store the fisher vector---%
                if ~isdir('finalFea\')
                    mkdir('finalFea\');
                end
                if iSam <= numSam_train
                    feaName = ['FV_feaTr' num2str(iSam) '_seg' num2str(iSeg) '_iBlock' num2str(iBlock)];
                else
                    feaName = ['FV_feaTs' num2str(iSam-numSam_train) '_seg' num2str(iSeg) '_iBlock' num2str(iBlock)];
                end
                save(['finalFea\',feaName,'.mat'],'feaFV');
                
            end            
        end
    end
end
clear sample;
clear samCluster;

%----- training SVM -------%
samTrain = [];
disp('Porcessing all training samples......');
training_label_vector = double(lab_trn);
for i = 1:numSam_train
    i
    tempSamFea = [];
    for iSeg = 1:1%====================
        for iBlock = 1:numBlock
            feaName = ['FV_feaTr' num2str(i) '_seg' num2str(iSeg) '_iBlock' num2str(iBlock)];
            load(['finalFea\',feaName,'.mat'],'feaFV');
            tempSamFea = [tempSamFea; feaFV];
        end
    end
    tempSamFea = tempSamFea';
    samTrain = [samTrain;tempSamFea];
end
training_instance_sparse = sparse(double(samTrain));
disp('Training svm model......');
svmModel = train(training_label_vector, training_instance_sparse);

samTest = [];
disp('Processing all testing samples......');
testing_label_vector = lab_tst;
for i = 1:numSam_test
    i
    tempSamFea = [];
    for iSeg = 1:1 %====================
        for iBlock = 1:numBlock
            feaName = ['FV_feaTs' num2str(i) '_seg' num2str(iSeg) '_iBlock' num2str(iBlock)];
            load(['finalFea\',feaName,'.mat'],'feaFV');
            tempSamFea = [tempSamFea; feaFV];
        end
    end
    tempSamFea = tempSamFea';
    samTest = [samTest;tempSamFea];
end
testing_instance_sparse = sparse(double(samTest));
disp('Testing......');
[predicted_label,accuracy,decision_values] = ...
    predict(testing_label_vector, testing_instance_sparse, svmModel);

save('samTest_s1.mat','samTest','-v7.3');
save('lab_trn_s1.mat','lab_trn','-v7.3');
save('lab_tst_s1.mat','lab_tst','-v7.3');
