clear all;

addpath(genpath('liblinear-2.1'));
addpath('general_function\');
addpath('libraryPubliced\tool\liblinear\');
addpath('libraryPubliced\tool\liblinear\matlab\');
%addpath('libraryPubliced\tool\libsvm\matlab\');

run('I:\wangyancheng\code\vlfeat-0.9.21\toolbox\vl_setup');

clusterfun = 'kmlim'; % or 'dplim', or 'no', or 'kmlim';
numClass = 10;
numCluster_list = [30 30 30 30 30 30 30 30 30 ];
numClSelect = 2;
numGmm = 4;
numBlock = 1;
gmmSamRatio = 0.6;

% seg_c = [1 2 3 4 5];
seg_c = 1% [1 2 3 4 5];

isPCA = 0;
isFVNorm = 0;
isFV = 0;
isMerge =0;

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
fileNames_Tr = [];

if isPCA  dPCA = 1024; else dPCA=dimension; end
sizeBlock = dPCA/numBlock;
new_path1 = 'D:\wangyancheng\code\ntudata';
if 1
    %--------- extract the data
    flag = [];
    
    if ~exist(new_path1)
        mkdir(new_path1); % 若不存在，在当前目录中产生一个子目录‘Figure’
    end
    fd= fopen('save_name.txt', 'w');
    %--------- extract the data
    for i = 1:length(index_s)-10
        i
        for j = 1:length(index_c)
            for k = 1:length(index_p)
                for m =1:length(index_r)
                    for d = 1:length(index_a)%-50
                        d
                        filename = ['S0',index_s(i,:), 'C0' ,index_c(j,:),'P0',index_p(k,:),'R0',index_r(m,:),'A0',index_a(d,:),'.mat'];
                        if exist([path filename],'file')
                        
                            if j>1
                                 fprintf(fd,'%s\r\n',filename);
                            fileNames_Tr = [fileNames_Tr;filename];
                            dt_train1 = load([path filename]);
                            data = reshape(cell2mat(dt_train1.data),[4096,5,11,2]);
                            data_train(:,:,:,idx_trn) = squeeze(data(:,:,:,1));
                            idx_trn = idx_trn+1;
                            lab_trn = [lab_trn;d];
                                
                                % in order to split the trainset
                                if(rand(1)>0.8)
                                    flag = [flag;1];
                                else
                                    flag = [flag;0];
                                end
                                
                            else
                                dt_test = load([path filename]);
                                data = reshape(cell2mat(dt_test.data),[4096,5,11,2]);
                                data_test(:,:,:,idx_tst) = squeeze(data(:,:,:,1));
                                lab_tst = [lab_tst;d];
                                idx_tst = idx_tst+1;
                            end
                        end
                    end
                end
            end
        end
    end
    fclose(fd);
end

    %------- PCA --------%
%     if isPCA
%         fprintf('----PCA----\n');
%         dt = reshape(data_train,[4096 5*11*size(data_train,4)]);
%         [pc,score,latent,tsquare] = pca(dt');
%         after_pca = score(:,1:dPCA);
%         
%         dt = reshape(data_test,[4096 5*11*size(data_test,4)]);
%         test_pca = dt' * pc(:,1:dPCA);
%         sample.train = reshape(after_pca',[dPCA 5 11 numSam_train]);
%         sample.test = reshape(test_pca',[dPCA 5 11 numSam_test]);
%         
%     end
    numSam_train = size(data_train,4);
    numSam_test = size(data_test,4);
    size(data_train)
    size(data_test)
    %     sample.all = cat(4,data_train,data_test);  % help cat  [4096 5 11 N]
    %
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
%     for i=1:size(data_train,2)
%         
%         filename1 = ['ntu_datamat_',num2str(i),'.mat'];
%         if ~exist([new_path1 '\' filename1])
%             
%             sub_sample = cat(3,squeeze(data_train(:,i,:,:)),squeeze(data_train(:,i,:,:)));
%             save([new_path1 '\' filename1],'sub_sample','lab_trn','numSam_train','numSam_test','flag','lab_tst','-v7.3');
%         end
%     end
%     clear data_train;
%     
%     clear data_test;
% end

% merge for segmation
se = 5;


%sample = Normlize(sample);
samp = [];
if isMerge
    for p = 2:4
        for d = (p+1):5
            p_max = sample.train(:,d,:,:)-sample.train(:,p,:,:);
            sample.train = [sample.train p_max];
            p_max = sample.test(:,d,:,:)-sample.test(:,p,:,:);
            sample.test = [sample.test p_max];
        end
    end
    se=9;
end
%     sample.train = samp(:,:,:,1:numSam_train);
%     sample.test = samp(:,:,:,numSam_train+1:end);
%     sample.all = [];
%     sample.all = samp;
%


% sample.train = sample.all(:,:,:,1:numSam_train);
% sample.test = sample.all(:,:,:,numSam_train+1:end);
% col = 11;row = 1;
% sam_instance = reshape(sample.all(:,:,:,1:numSam_train),[4096 size(sample.all,2) 11*numSam_train]);
% lab_instan = repmat(lab_trn,[1 11])';
% lab_instan = lab_instan(:);
%tst_instance = reshape(sample.all(:,:,:,numSam_train+1:end),[4096 size(sample.all,2) 11*numSam_test]);

%------ cluster patches of training sample in iLayer ------%
%samCluster_all = [reshape(sample.train,dPCA,55*numSam_train)]';

% Lab123 = repmat(lab_trn,1,11)';
% samImgLabel = Lab123(:);  %每个特征对应的标签
aaa = [];
for iSeg =1% seg_c%[1 6 7 8 9]%1:se
    iSeg
  
    numSam = numSam_train + numSam_test;
    
    samCluster = reshape(data_train(:,iSeg,:,:),[4096 11*numSam_train]);
    lab_instan = repmat(lab_trn,[1 11])';
    lab_instan = lab_instan(:);
    samCluster = samCluster'; %[N1*11 4096]
    
    if ~strcmp(clusterfun,'no')&&(numCluster_list(iSeg)~= 0)
        
        numCluster = numCluster_list(iSeg);
        if strcmp(clusterfun,'dplim')
            res_cluster = func_dpCluster(samCluster,numCluster);
        end
        if strcmp(clusterfun,'kmlim')
            new_path2 = 'D:\wangyancheng\code\ntu\cluster_data';
            if ~exist(new_path2)
                mkdir(new_path2); % 若不存在，在当前目录中产生一个子目录‘Figure’
            end
            
            filename = ['k' num2str(numCluster) '_s' num2str(iSeg) '.mat'];
            if ~exist([new_path2 '\' filename])
                
                
                res_cluster = func_kmCluster(samCluster,numCluster,1000,1);
                save([new_path2 '\' filename],'res_cluster');
            else
                load([new_path2 '\' filename])
            end
        end
        cluster_Label = res_cluster.labels;
       
        %% find the cluster center sample
        if 1
            dSam=samCluster;%[numSam,~]
            cluster_center =res_cluster.centersFea;%[numCenter,~]
            samDist = func_distCenter(dSam,cluster_center); %NxC
            for i = 1:30
                 cluster_indx = find(samDist(:,i) == min(samDist(:,i)), 1 );
                 file_idx = floor(cluster_indx/11);
                 view_idx = mod(cluster_indx,11)
                 fileNames_Tr(file_idx,:)
                 disp('------------------');
            end
            
        end
        
        
        
        
        %%
        for i = 1:numCluster
            numInCluster(i) = length(find(cluster_Label == i));  %各个簇所含特征数量
            for j = 1:numClass
                idxClassJ = find(lab_instan == j);                       %标签j在所有特征中的位置
                clusterLabelForClassJ = cluster_Label(idxClassJ);     %标签j下的特征所属簇    找到标签j的簇label
                numEachClInClass(i,j) = length(find(clusterLabelForClassJ == i)); %第i个簇中第j标签的特征数量
            end
            ratioEachCl(i,:) = numEachClInClass(i,:) / numInCluster(i);  % 每个簇在不同类别下的分布
            ratioEachCl(i,ratioEachCl(i,:)==0 ) = 0.0001;
            globalEntropy(i) = 1 + sum(ratioEachCl(i,:).*log(ratioEachCl(i,:)))/log(numClass);
        end
        
        % split the train instance and lab into train set & test set for 1st stage
      %  train_instan = samCluster;
        %tst_instance = sparse(squeeze(tst_instance(:,iSeg,:))');
        
        flag_ = repmat(flag,[1 11])';
        flag_ = flag_(:);
        test_instance = samCluster(flag_==1,:);
        train_instance = samCluster(flag_==0,:);
        test_lab = lab_instan(flag_==1);
        lab_instance = lab_instan(flag_==0);
        
        
        % 1st stage to find the mis-predicted cluster instance
        tic
        samGmm = samCluster;%[N1*11 4096]
        disp('Computing the gmm model......');
        [gmmComp.mean,gmmComp.covariances,gmmComp.priors] = vl_gmm(samGmm',numGmm);toc;
        disp('stage1:gmm have been learned!');
        %----- FV coding -----%
        tic;
        stage1_feaFV = zeros(4096*numGmm*2,numSam_train);
        for iSam = 1:numSam_train
            dSam = squeeze(data_train(:,iSeg,:,iSam))';
            stage1_feaFV(:,iSam) = vl_fisher(dSam',gmmComp.mean,gmmComp.covariances,gmmComp.priors,'Improved');
        end
        toc;
        disp('stage1:fv have been computed!');
        stage1_train_fv = stage1_feaFV(:,flag==0);
        stage1_test_fv = stage1_feaFV(:,flag==1);
        stage1_train_label = lab_trn(flag==0);
        stage1_test_label = lab_trn(flag==1);
        stage1_svm_model = train(double(stage1_train_label), sparse(stage1_train_fv'));
        [pred,acc,dec] = predict(double(stage1_test_label), sparse(stage1_test_fv'), stage1_svm_model);
        
        %clear stage1_test_fv stage1_train_fv stage1_train_label stage1_svm_model
        
        for ci = 1:numClass
            stage1_num_in_class(ci) = length(find(stage1_test_label==ci));
            for cj = 1:numClass
                %                 c_end = sum(stage1_num_in_class(1:ci));
                %                 c_start = c_end - stage1_num_in_class(ci)+1;
                c_r = find(stage1_test_label==ci);
                stage1_confusion_matrix(ci,cj)=length(find(pred(c_r)==cj))/stage1_num_in_class(ci);
            end % 找到每个类别容易被预测成的多个类别
            class_acc(ci) =  stage1_confusion_matrix(ci,ci);
        end
        
        alpha = 0.05;
        
        
        for i = 1:numClass
            
            [label_Class,idxsort] = sort(stage1_confusion_matrix(i,:),'descend');
            
            %        localClass = find(stage1_confusion_matrix(i,:)>alpha);
            %         if length(localClass)>2
            localClass = idxsort(1:2);
            %         end
            for k = 1:numCluster
                
                ratioLocalCl = numEachClInClass(k,localClass) / numInCluster(k);
                ratioLocalCl(ratioLocalCl==0) = 0.0001;
                ratioLocalCl = ratioLocalCl/sum(ratioLocalCl);
                localEntropy(i,k) = 1 + sum(ratioLocalCl.*log2(ratioLocalCl))/log2(length(localClass));
                
            end
            
        end
        
        %stage 2
        beta = 0;
        gamma = 0.016;
        
        idx_discard = [];
        length_ = 0;
        [class_acc_rank,idx_rank] = sort(class_acc);
        score = localEntropy;
        for i = 1:numClass
            len = length(find(lab_instan==i));
            if class_acc(i)<0.95
                %score(i,:) = globalEntropy.*(beta+localEntropy(i,:));
                %         score(i,:) = globalEntropy;
                %         gamma = 0.004 ;% (1-class_acc(i))/15;
                
                %         gamma = 0.025 - find(idx_rank==i)/400;
                %         idxlocal = find(localEntropy(i,:)<gamma);
                %         idx_c = idxlocal(globalEntropy(idxlocal)<0.1);
                score(i,:) = (1-beta)*globalEntropy + beta*localEntropy(i,:);
                %         [scorePattern idxscore] = sort(Score(i,:));
                
                %      idx_c = find(score(i,:) <gamma);  %找到得分小于阈值的pattern
                [a,id] = sort(score(i,:));
                idx_c = id(1);
                if ~isempty(idx_c)
                    idx_dis = [];
                    for k = 1:length(idx_c)
                        idx = find(cluster_Label==idx_c(k));
                        idx_dis = [idx_dis;idx];   %找到sam_instance中最终要丢弃的instance
                    end
                    idx_dis = idx_dis(idx_dis>length_+1 & idx_dis<length_+len);
                    idx_discard = [idx_discard; idx_dis'];
                end
            end
            length_ = length_+len;
        end
        
        aaa(iSeg) = length(idx_discard)
        zz = randperm(length(idx_discard));
        %idx_discard = idx_discard(zz(1:length(idx_discard)/2))';
        
        %%%%%%%%%%%%%% used to select sample for GMM
        
        %------ select patches ------%
        samGmm = samCluster;
        samGmm(idx_discard,:) = [];
        clear samCluster;
        %------ learning gmm ------%
        
        disp('Computing the gmm model......');
        [gmmComp.mean,gmmComp.covariances,gmmComp.priors] = vl_gmm(samGmm',numGmm);%[4096 N1*11]
        if ~isdir('gmmComp\')
            mkdir('gmmComp\');
        end
        save(['gmmComp\' 'resGmm_' num2str(iSeg) '.mat'],'gmmComp');
        disp('gmmComp have been saved!');
        clear samGmm
        %----- FV coding -----%
        b = ones(numSam_train*11,1);
        b(idx_discard) = 0;
        b = reshape(b,11,numSam_train)';
        for iSam = 1:numSam_train
            dSam = squeeze(data_train(:,iSeg,:,iSam))';
            dSam(b(iSam,:)==0,:)=[];
            if isFVNorm
                feaFV = vl_fisher(dSam',gmmComp.mean,gmmComp.covariances,gmmComp.priors,'Normalized');
            else
                feaFV = vl_fisher(dSam',gmmComp.mean,gmmComp.covariances,gmmComp.priors,'Improved');
            end
            %-- store the fisher vector---%
            if ~isdir('finalFeaTR\')
                mkdir('finalFeaTR\');
            end
            feaName = ['FV_feaTr' num2str(iSam) '_seg' num2str(iSeg)];
            save(['finalFeaTR\',feaName,'.mat'],'feaFV');
            disp(['Got ',num2str(iSam),'/',num2str(numSam),'--',num2str(iSeg),num2str(se) ' samples' ' final feature data .....']);
        end
        
        for iSam= 1:numSam_test
            
            if ~isdir('finalFeaTs\')
            mkdir('finalFeaTs\');
            end
            dSam = squeeze(data_test(:,iSeg,:,iSam))';
            feaFV = vl_fisher(dSam',gmmComp.mean,gmmComp.covariances,gmmComp.priors,'Improved');
             feaName = ['FV_feaTs' num2str(iSam) '_seg' num2str(iSeg)];
             save(['finalFeaTs\',feaName,'.mat'],'feaFV');
        end
        
    else
%         samGmm = samCluster;%[N1*11 4096]
%         clear samCluster;
        %------ learning gmm ------%
        samgmm = samCluster';%[4096 N1*11]
      
        disp('Computing the gmm model......');
        [gmmComp.mean,gmmComp.covariances,gmmComp.priors] = vl_gmm(samgmm,numGmm);
        if ~isdir('gmmComp\')
            mkdir('gmmComp\');
        end
        %%save(['gmmComp\' 'resGmm_' num2str(iSeg) '.mat'],'gmmComp');
        %disp('gmmComp have been saved!');
        disp('gmm have been kearned!');
        %----- FV coding -----%
        for iSam = 1:numSam_train
            dSam = squeeze(data_train(:,iSeg,:,iSam))';
            
            if isFVNorm
                feaFV = vl_fisher(dSam',gmmComp.mean,gmmComp.covariances,gmmComp.priors,'Normalized');
            else
                feaFV = vl_fisher(dSam',gmmComp.mean,gmmComp.covariances,gmmComp.priors,'Improved');
            end
            %-- store the fisher vector---%
            if ~isdir('finalFeaTR\')
                mkdir('finalFeaTR\');
            end
            feaName = ['FV_feaTr' num2str(iSam) '_seg' num2str(iSeg)];
            save(['finalFeaTR\',feaName,'.mat'],'feaFV');
            disp(['Got ',num2str(iSam),'/',num2str(numSam),'--',num2str(iSeg),num2str(se) ' samples' ' final feature data .....']);
        end
        
        for iSam= 1:numSam_test
            
            if ~isdir('finalFeaTs\')
            mkdir('finalFeaTs\');
            end
            dSam = squeeze(data_test(:,iSeg,:,iSam))';
            feaFV = vl_fisher(dSam',gmmComp.mean,gmmComp.covariances,gmmComp.priors,'Improved');
             feaName = ['FV_feaTs' num2str(iSam) '_seg' num2str(iSeg)];
             save(['finalFeaTs\',feaName,'.mat'],'feaFV');
        end
        
        
    end
    clear sub_sample
end

clear feaName feaFV dsamgmm gmmComp samgmm b


%----- training SVM -------%
samTrain = [];
disp('Porcessing all training samples......');
training_label_vector = double(lab_trn);
for i = 1:numSam_train
    i
    tempSamFea = [];
    for iSeg = seg_c
        for iBlock = 1:numBlock
            feaName = ['FV_feaTr' num2str(i) '_seg' num2str(iSeg)];
            load(['finalFeaTR\',feaName,'.mat'],'feaFV');
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
    for iSeg = seg_c%1:se
        for iBlock = 1:numBlock
            feaName = ['FV_feaTs' num2str(i) '_seg' num2str(iSeg)];
            load(['finalFeaTs\',feaName,'.mat'],'feaFV');
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

% predict_label= predicted_label;
% 
% for i = 1:numClass
%     num_in_class(i) = length(find(lab_tst==i));
%     
% end
% name_class = importdata('classInd.txt');
% for ci = 1:numClass
%     for cj = 1:numClass
%         c_end = sum(num_in_class(1:ci));
%         c_start = c_end - num_in_class(ci)+1;
%         confusion_matrix(ci,cj)=length(find(predict_label(c_start:c_end)==cj))/num_in_class(ci);
%         
%     end
% end
% 
% draw_cm(confusion_matrix,name_class,numClass);

