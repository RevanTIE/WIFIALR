clc
clear

dat_orig = load('../trn_tst/X.csv');
[sizerow, sizecol] = size(dat_orig);
lbl_orig = load('../trn_tst/Y.csv');

%% Se deben revolver los datos
idx = randperm(sizerow);
datos = dat_orig; %Vector de X
labels = lbl_orig; %Vector de Y

datos(idx,1:end) = dat_orig(:,1:end);
labels(idx,1) = lbl_orig(:,1);

trn_percent = [0.6, 0.7, 0.8, 0.9];
trn_x100 = 100 * trn_percent;
for t = 1:length(trn_percent)
    %% TRAINING %%
    training_percent = trn_percent(1, t);
    TRN_PERCENT =  floor(sizerow * trn_percent(1, t));
    TRN_Y = labels((1:TRN_PERCENT), :);
    TRN_SET = datos((1:TRN_PERCENT), :);

    %% TEST %%
    TST_Y = labels((TRN_PERCENT+1:end), :);
    TST_SET = datos((TRN_PERCENT+1:end), :);

    K = 5; %% Vecinos
    predict_KNN = [];

    %KNN
    Modelo_KNN = fitcknn(TRN_SET, TRN_Y,'NumNeighbors',K);
    predict_KNN = predict(Modelo_KNN, TST_SET);
    
    tasa_KNN(t) = (sum(predict_KNN==TST_Y)/length(TST_Y))* 100;
end

plot(trn_x100,tasa_KNN','-ob', 'DisplayName', 'KNN');legend();hold on;
xlabel('% TRN');
ylabel('% Tasa de reconocimiento');
hold off;



