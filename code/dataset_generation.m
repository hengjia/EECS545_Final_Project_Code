clear all;
clc;

% Load the data
TrainFile = 'train-images.idx3-ubyte';
TrainLabel = 'train-labels.idx1-ubyte';
fid = fopen(TrainFile,'r');
a = fread(fid,16,'uint8');
MagicNum = ((a(1)*256+a(2))*256+a(3))*256+a(4);
ImageNum = ((a(5)*256+a(6))*256+a(7))*256+a(8);
ImageRow = ((a(9)*256+a(10))*256+a(11))*256+a(12);
ImageCol = ((a(13)*256+a(14))*256+a(15))*256+a(16);
training_data = zeros(ImageNum,ImageRow*ImageRow);
training_label = zeros(ImageNum,1);
for i=1:ImageNum
    b = fread(fid,ImageRow*ImageCol,'uint8');   
    c = reshape(b,[ImageRow ImageCol]); 
    d = c'./255; 
    training_data(i,:)=reshape(d,[1,ImageCol*ImageRow]);
end
fclose(fid);
fid = fopen(TrainLabel,'r');
a = fread(fid,8,'uint8');
for i=1:ImageNum
    b = fread(fid,1,'uint8');   
    training_label(i)=double(b);
end
fclose(fid);

% Sort the data
[~, idx] = sort(training_label); % Get the increasing order of training label
training_label = training_label(idx);
training_data=training_data(idx,:);

num_label = zeros(10,1);
for i=1:10
    num_label(i)=sum(training_label==(i-1));
end
temp_data = zeros(50000,784);
temp_label = zeros(50000,1);
temp_data(1:5000,:) = training_data(1:5000,:);temp_label(1:5000,:) = training_label(1:5000,:);
temp_data(5001:10000,:) = training_data(5924:10923,:);temp_label(5001:10000) = training_label(5924:10923);
temp_data(10001:15000,:) = training_data(12666:17665,:);temp_label(10001:15000) = training_label(12666:17665);
temp_data(15001:20000,:) = training_data(18624:23623,:);temp_label(15001:20000) = training_label(18624:23623);
temp_data(20001:25000,:) = training_data(24755:29754,:);temp_label(20001:25000) = training_label(24755:29754);
temp_data(25001:30000,:) = training_data(30597:35596,:);temp_label(25001:30000) = training_label(30597:35596);
temp_data(30001:35000,:) = training_data(36018:41017,:);temp_label(30001:35000) = training_label(36018:41017);
temp_data(35001:40000,:) = training_data(41936:46935,:);temp_label(35001:40000) = training_label(41936:46935);
temp_data(40001:45000,:) = training_data(48201:53200,:);temp_label(40001:45000) = training_label(48201:53200);
temp_data(45001:50000,:) = training_data(54052:59051,:);temp_label(45001:50000) = training_label(54052:59051);

training_data = temp_data;
training_label = temp_label;
save hand_written_digits_545.mat training_data training_label