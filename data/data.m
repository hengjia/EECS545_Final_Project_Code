clc;clear;close all;
load hand_written_digits_545.mat; 
select_4 = (training_label == 4);
select_9 = (training_label == 9);
X = [training_data(select_4,:);training_data(select_9,:)];
Y = [ones(5000,1); -ones(5000,1)];