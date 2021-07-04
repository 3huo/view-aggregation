clear all;

addpath(genpath('liblinear-2.1'));
addpath('general_function\');
addpath('libraryPubliced\tool\liblinear\');
addpath('libraryPubliced\tool\liblinear\matlab\');
%addpath('libraryPubliced\tool\libsvm\matlab\');


run('I:\wangyancheng\code\vlfeat-0.9.21\toolbox\vl_setup');

clusterfun = 'kmlim'; % or 'dplim', or 'no', or 'kmlim';
numClass = 10;
numCluster = 30;
numGmm = 1;
numBlock = 1;
gmmSamRatio = 0.3;

isPCA = 0;
isFVNorm = 0;
isFV = 1;

path = 'I:\wangyancheng\feature_ucla_share_MBB\';% data path
index_a = ['01';'02';'03';'04';'05';'06';'08';'09';'11';'12'];
index_s = ['01';'02';'03';'04';'05';'06';'07';'08';'09';'10'];
index_e = ['00';'01';'02';'03';'04';'05';'06';'07';'08';'09';'10'];%
index_v = ['01';'02';'03'];
%data_train = zeros(4096,5,11,length(dir([path '*_v01*']))+length(dir([path '*_v02*'])));
data_train = [];
data_test = [];
idx_trn =1;
idx_tst = 1;
lab_trn = [];
lab_tst = [];
dimension = 4096;
dPCA = 4096;
sizeBlock = dPCA/numBlock;
%--------- extract the data
for i = 1:length(index_a)
    i
    for j = 1:length(index_s)
        for k = 1:length(index_e)
            for m =1:length(index_v)
                filename = ['a',index_a(i,:), '_s' ,index_s(j,:),'_e',index_e(k,:),'_v',index_v(m,:),'.mat'];
                if exist([path filename],'file')
                    if m<3
                        dt_train1 = load([path filename]);
                        data = reshape(cell2mat(dt_train1.data),[4096,5,11,2]);
                        data_train(:,:,:,idx_trn) = squeeze(data(:,:,:,1));
                        idx_trn = idx_trn+1;
                        lab_trn = [lab_trn;i];
                    else
                        dt_test = load([path filename]);
                        data = reshape(cell2mat(dt_test.data),[4096,5,11,2]);
                        data_test(:,:,:,idx_tst) = squeeze(data(:,:,:,1));
                        lab_tst = [lab_tst;i];
                        idx_tst = idx_tst+1;
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

            
            col = 11;row = 1;
%------ cluster patches of training sample in iLayer ------%
%samCluster_all = [reshape(sample.train,dPCA,55*numSam_train)]';

Lab123 = repmat(lab_trn,1,11)';
samImgLabel = Lab123(:);  %每个特征对应的标签
for iSeg = 1:5
    samCluster = [reshape(sample.train(:,iSeg,:,:),dPCA,11*numSam_train)]'; %[N1*11 4096]
    
    if ~strcmp(clusterfun,'no')
        if strcmp(clusterfun,'dplim')
            res_cluster = func_dpCluster(samCluster,numCluster);
        end
        if strcmp(clusterfun,'kmlim')
            res_cluster = func_kmCluster1(samCluster,numCluster,10000,5);
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
            save(['sampleforFV\',fvSamName,'_ucla.mat'],'fvSam');
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
%                 feaName = ['FV' num2str(iSam) '_seg' num2str(iSeg)];
%                 save(['FV_feaTs\',feaName,'.mat'],'feaFV');

                
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
%                 feaName = ['FV' num2str(iSam) '_seg' num2str(iSeg)];
%                 save(['FV_feaTs\',feaName,'.mat'],'feaFV');

            end            
        end
    end
end
%----- training SVM -------%
samTrain = [];
disp('Porcessing all training samples......');
training_label_vector = double(lab_trn);
for i = 1:numSam_train
    tempSamFea = [];
    for iSeg = 1:5
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
    tempSamFea = [];
    for iSeg = 1:5
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

