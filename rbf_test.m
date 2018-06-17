clear all
close all
clc

%% Read data

train_file = 'train.csv';
test_file = 'test.csv';

train_database = csvread(train_file, 1);
test_database = csvread(test_file, 1);

%% Split train-validation

entries = size(train_database,1);

validation_database = train_database([1:(entries/3)],:);
train_database = train_database([(entries/3)+1:end],:);

%% Normalization

train_min = max(max(train_database));
train_max = min(min(train_database));

train_database(:,[1:end-1]) = (2*((train_database(:,[1:end-1]) - train_min)/(train_max - train_min))) - 1;
validation_database(:,[1:end-1]) = (2*((validation_database(:,[1:end-1]) - train_min)/(train_max - train_min))) - 1;
test_database(:,[1:end-1]) = (2*((test_database(:,[1:end-1]) - train_min)/(train_max - train_min))) - 1;
%% Create RBF

confusion_matrix = zeros(3,3);
MaxHits = 0;
MaxNeurons = 0;
MaxSpread = 0;

%Train 2
spread = 11.18;
maxNeurons = 6;

error_goal = 0.0;
DF = 1;

i = 1;

features_train = train_database(:,[1:end-1])';
class_train = train_database(:,end)';

rbf_network = newrb(features_train,class_train, error_goal, spread,maxNeurons,DF);
close all;

%% Test RBF
features_test = test_database(:,[1:end-1])';

prediction_test = rbf_network(features_test);
prediction_test = round(prediction_test');

class_test = test_database(:,end);

for i = 1:size(class_test,1)
    confusion_matrix(prediction_test(i)+1,class_test(i)+1) = confusion_matrix(prediction_test(i)+1,class_test(i)+1) + 1;
end

confusion_matrix
%sprintf('Hits: %f.',(hitsTest/size(class_test,1))*100)
