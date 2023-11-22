close all
clear variables

% Load the flower images into a dataset

flowersDataset = fullfile("17flowers/");

% Images are stored in a folder with their label type
imds = imageDatastore(flowersDataset, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

numTrainFiles = 80 * 0.8;
[trainingSet,validationSet] = splitEachLabel(imds,numTrainFiles,'randomized');

countEachLabel(trainingSet)
countEachLabel(validationSet)

trainingSet.ReadSize = 6;
validationSet.ReadSize = 6;

trainingSet = transform(trainingSet, @imageAug, 'IncludeInfo',true);
validationSet = transform(validationSet, @imageAug, 'IncludeInfo',true);

figure;
ims = preview(trainingSet);
ims = ims(:,1);
montage(ims(:));
title("Augmented Training Set Image");

figure;
ims = preview(validationSet);
ims = ims(:,1);
montage(ims(:));
title("Augmented Validation Set Image");

% trainingSet.UnderlyingDatastores{1}.ReadSize = 1;
% validationSet.UnderlyingDatastores{1}.ReadSize = 1;


inputSize = [256 256 3];
numClasses = 17;

layers = [
    imageInputLayer(inputSize)

    convolution2dLayer(4,64)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'stride', 2)

    convolution2dLayer(4,128)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'stride', 2)

    convolution2dLayer(4,256)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'stride', 2)

    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer
    ]

options = trainingOptions('sgdm', ...
    'MiniBatchSize',64,...
    'MaxEpochs',30, ...
    'InitialLearnRate',0.001,...
    'ValidationData',validationSet, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress', ...
    'ExecutionEnvironment','multi-gpu')


classifier = trainNetwork(trainingSet, layers,options);

%save('classnet.mat', 'classifier')

% load('classnet.mat', 'classifier')

% deepNetworkDesigner(classifier);

predictedLabels = predict(classifier, validationSet, 'ReturnCategorical', true);
predictedTrainingLabels = predict(classifier, trainingSet, 'ReturnCategorical', true);

disp("Validation Data")

figure;
confMat = confusionmat(validationSet.UnderlyingDatastores{1}.Labels, predictedLabels);
helperDisplayConfusionMatrix(confMat)
confusionchart(confMat, unique(validationSet.UnderlyingDatastores{1}.Labels))
title("Validation Set Prediction/True Classes");

accuracy = mean(predictedLabels == validationSet.UnderlyingDatastores{1}.Labels)
precision = mean(diag(confMat) ./ sum(confMat,2));
recall = mean(diag(confMat) ./ sum(confMat,1)');
F1score = 2 * (precision * recall) / (precision + recall);

precision
recall
F1score


disp("Training Data")

figure;
confMat = confusionmat(trainingSet.UnderlyingDatastores{1}.Labels, predictedTrainingLabels);
helperDisplayConfusionMatrix(confMat)
confusionchart(confMat, unique(trainingSet.UnderlyingDatastores{1}.Labels))
title("Training Set Prediction/True Classes");

accuracy = mean(predictedTrainingLabels == trainingSet.UnderlyingDatastores{1}.Labels)
precision = mean(diag(confMat) ./ sum(confMat,2));
recall = mean(diag(confMat) ./ sum(confMat,1)');
F1score = 2 * (precision * recall) / (precision + recall);

precision
recall
F1score

function helperDisplayConfusionMatrix(confMat)
% Display the confusion matrix in a formatted table.

% Convert confusion matrix into percentage form
confMat = bsxfun(@rdivide,confMat,sum(confMat,2));

digits = 1:17;
colHeadings = arrayfun(@(x)sprintf('%d',x),0:16,'UniformOutput',false);
format = repmat('%-9s',1,11);
header = sprintf(format,'digit  |',colHeadings{:});
fprintf('\n%s\n%s\n',header,repmat('-',size(header)));
for idx = 1:numel(digits)
    fprintf('%6d%-s',digits(idx),  ' | ');
    fprintf('%-9.2f', confMat(idx,:));
    fprintf('\n')
end
end

function [dataOut,info] = imageAug(dataIn,info)
dataOut = cell([size(dataIn,1),2]);


for i = 1:size(dataIn,1)
    temp = dataIn{i};
    
    temp = imresize(temp, [256, 256]);

    tform = randomAffine2d(Scale=[0.95,1.05],Rotation=[-45 45],XShear=[-10 10],YShear=[-10 10]);
    outputView = affineOutputView(size(temp),tform);
    temp = imwarp(temp,tform,OutputView=outputView);

    dataOut(i,:) = {temp,info.Label(i)};
end

end

