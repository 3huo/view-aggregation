clear all;

addpath(genpath('liblinear-2.1'));
addpath('general_function\');
addpath('libraryPubliced\tool\liblinear\');
addpath('libraryPubliced\tool\liblinear\matlab\');
%addpath('libraryPubliced\tool\libsvm\matlab\');


run('I:\wangyancheng\code\vlfeat-0.9.21\toolbox\vl_setup');

numClass = 60;


path = 'G:\wangyancheng\ntu_rgbd\dynamic\feature_ntu_subject_share_MBB\';% data path
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
view_trn = [];
view_tst = [];
dimension = 4096;
All_fileName_tr = [];
All_fileName_ts = [];
%--------- extract the data
for i = [1,2,3,4,5,6,7]%1:length(index_s)
    i
    for j = 1:length(index_c)
        
        for k = 1:length(index_p)
            for m =1:length(index_r)
                for d = 1:length(index_a)
                    filename = ['S0',index_s(i,:), 'C0' ,index_c(j,:),'P0',index_p(k,:),'R0',index_r(m,:),'A0',index_a(d,:),'.mat'];
                    if exist([path filename],'file')
                        
                        if  k<21
                            All_fileName_tr = [All_fileName_tr;filename];
                            dt_train1 = load([path filename]);
                            data = reshape(cell2mat(dt_train1.data),[4096,5,11,2]);
                            data_train(:,:,:,idx_trn) = squeeze(data(:,:,:,1));
                            idx_trn = idx_trn+1;
                            lab_trn = [lab_trn;d];
                            view_trn = [view_trn;j];
                        else
                            All_fileName_ts = [All_fileName_ts;filename];
                            dt_test = load([path filename]);
                            data = reshape(cell2mat(dt_test.data),[4096,5,11,2]);
                            data_test(:,:,:,idx_tst) = squeeze(data(:,:,:,1));
                            lab_tst = [lab_tst;d];
                            idx_tst = idx_tst+1;
                            view_tst = [view_tst;j];
                        end
                    end
                end
            end
        end
    end
end


max_pooling_tr = [];
average_pooling_tr = [];
max_pooling_ts = [];
average_pooling_ts = [];

for iSeg = 1%:5
    %train
    samCluster = squeeze(data_train(:,iSeg,:,:)); %[4096 11 N]
    max_p = squeeze(max(samCluster,[],2));
    avg_p = squeeze(mean(samCluster,2));
    max_pooling_tr = [max_pooling_tr;max_p];
    average_pooling_tr = [average_pooling_tr;avg_p];
    %test
    samCluster = squeeze(data_test(:,iSeg,:,:)); %[4096 11 N]
    max_p = squeeze(max(samCluster,[],2));
    avg_p = squeeze(mean(samCluster,2));
    max_pooling_ts = [max_pooling_ts;max_p];
    average_pooling_ts = [average_pooling_ts;avg_p];
end


%% correlation 
% label
concatenation_train = reshape(data_train(:,1,:,:),4096*11,size(data_train,4));
concatenation_test = reshape(data_test(:,1,:,:),4096*11,size(data_test,4));
r_c = [];
r_pm = [];
r_pa = [];
for i = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,22,20,21,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,56,55,57,58,59,60]
   i
% view2 --train
idx_a = find(lab_trn==i);
a_name = All_fileName_tr(idx_a,:);

idx_v2 = find(view_trn==2);
v2_name = All_fileName_tr(idx_v2,:);
idx = intersect(idx_a,idx_v2);
concatenation_v2 = concatenation_train(:,idx);
m_pooling_v2 = max_pooling_tr(:,idx);
a_pooling_v2 = average_pooling_tr(:,idx);
len2 = length(idx)
% view 3 --train
idx_v3 = find(view_trn==3);
idx = intersect(idx_a,idx_v3);
concatenation_v3 = concatenation_train(:,idx);
m_pooling_v3 = max_pooling_tr(:,idx);
a_pooling_v3 = average_pooling_tr(:,idx);
len3 = length(idx)
%view1 --test
idx_a = find(lab_tst==i);
idx_v1 = find(view_tst==1);
idx = intersect(idx_a,idx_v1);
concatenation_v1 = concatenation_test(:,idx);
m_pooling_v1 = max_pooling_ts(:,idx);
a_pooling_v1 = average_pooling_ts(:,idx);
len1 = length(idx)
len = min([len1,len2,len3])
%===cor compute
r = corrcoef(concatenation_v1(:,1:len),concatenation_v3(:,1:len))
r_c = [r_c; r(1,2)];
r = corrcoef(m_pooling_v1(:,1:len),m_pooling_v3(:,1:len));
r_pm = [r_pm; r(1,2)];
r = corrcoef(a_pooling_v1(:,1:len),a_pooling_v3(:,1:len));
r_pa = [r_pa; r(1,2)];

end
mean(r_c)
mean(r_pm)
mean(r_pa)

