clear all
close all
clc

%% Read data

train_file = 'train.csv';

train_database = csvread(train_file, 1);

%% Split train-validation

entries = size(train_database,1);

validation_database = train_database([1:(entries/3)],:);
train_database = train_database([(entries/3)+1:end],:);

%% Normalization

train_min = max(max(train_database));
train_max = min(min(train_database));

train_database(:,[1:end-1]) = (2*((train_database(:,[1:end-1]) - train_min)/(train_max - train_min))) - 1;
validation_database(:,[1:end-1]) = (2*((validation_database(:,[1:end-1]) - train_min)/(train_max - train_min))) - 1;
%% Create RBF

historyHits = [];
MaxHits = 0;
MaxNeurons = 0;
MaxSpread = 0;

%Train
spread = 11.18;
maxNeurons = 6;

error_goal = 0.0;
DF = 1;

%Change number of neurons
for maxNeurons = 1:1:size(train_database(:,[1:end-1])',2)

    features_train = train_database(:,[1:end-1])';
    class_train = train_database(:,end)';

    rbf_network = newrb(features_train,class_train, error_goal, spread,maxNeurons,DF);
    close all;
    
    %% Validate RBF  
    features_validation = validation_database(:,[1:end-1])';

    prediction_validation = rbf_network(features_validation);
    prediction_validation = round(prediction_validation');

    class_validation = validation_database(:,end);

    hitsValidation = 0;
    for i = 1:size(class_validation,1)
        if prediction_validation(i) == class_validation(i)
            hitsValidation = hitsValidation + 1;
        end
    end
    
    historyHits(end+1) = (hitsValidation/size(class_validation,1))*100;
    
    if MaxHits < hitsValidation
        MaxHits = hitsValidation;
        MaxNeurons = maxNeurons;      
    end
end

plot(historyHits, 'r', 'LineWidth',2);
title('Acertos X Neurônios');
xlabel('Quantidade de neurônios');
ylabel('Taxa de acerto (%)');

sprintf('Neurons %f\nHits: %f.',MaxNeurons, (MaxHits/size(class_validation,1))*100)
