clc;
clear;
dat_orig = load('../trn_tst/X.csv');
[sizerow, sizecol] = size(dat_orig);
lbl_orig = load('../trn_tst/Y.csv');

%% Se deben revolver los datos
idx = randperm(sizerow);
datos = dat_orig; %Vector de X
labels = lbl_orig; %Vector de Y

datos(idx,1:end) = dat_orig(:,1:end);
labels(idx,1) = lbl_orig(:,1);

%PLANTILLA DE UNA SVM
%KERNELES= gaussian, linear, polynomial
%PARÁMETRO DE COMPLEJIDAD C=2^-5 2^4 ...2^5
%PARA EL KERNEL POLINOMIAL USAR EL ORDEN DEL POLINOMIO d=2,3,4,5
%PARA EL KERNEL GAUSSIANO MODIFICAR EL ANCHO DEL KERNEL GAMMA=2^-5 2^4 ...2^5

t = templateSVM('Standardize',true,...          %ESTANDARIZAR ENTRADAS
                'BoxConstraint',1,...           %          C
                'KernelFunction','linear'); %,...%   KERNEL
                %'PolynomialOrder',2);          %          d
                %'KernelScale',2^5);            %        GAMMA
                

%CREACION DEL MODELO MULTICLASE. TIPOS: onevsone,  onevsall
Model = fitcecoc(datos,labels,'Learners',t,...
                              'Coding','onevsall',...
                              'KFold',10',...
                              'Verbose',1);

%OBTENCION DEL ERROR DE CROSSVALIDATION
Error_de_CV = kfoldLoss(Model);

fprintf('Error de crossvalidación=%f\n',Error_de_CV);
fprintf('Exactitud de crossvalidación=%f\n',1-Error_de_CV);
