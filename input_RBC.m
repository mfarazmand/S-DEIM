%%
clc; clear;

%% parameters
r=25;  % number of sensors
m=50; % number of modes

%% Compute POD modes
load RBC_data_POD.mat % load data for computing POD modes
meanX = mean(X_POD,2); % compute the mean
[Phi,M,~]=svd(X_POD-meanX,'econ');
Phi=Phi(:,1:m); % extract m POD modes

%% Compute sensor locations
[~,~,pivot] = qr(Phi','vector');
ind = pivot(1:r);
N = size(X_POD,1);
S=eye(N);
S=S(:,ind);

%% prepare training data
load RBC_data_train.mat
y_tr = S'*(X_tr-meanX); % centered observations
Z = null(S'*Phi); % null space
xi_tr = Z'*Phi'*(X_tr-meanX); % optimal kernel vector

%% RC - train
rng(6,'twister')
n_res=500; % size of the reservoir
A = sprand(n_res,n_res,.1);
A = full(A); % reservoir matrix
ind = (A~=0); A(ind) = A(ind)-.5; % make sure nonzero entries are within (-0.5,0.5)
A = 0.99*A/max(abs(eig(A))); % rescale to obtain spectral radius<1
b = rand(n_res,1)-.5; % biases
Win = rand(n_res,r)-.5; % input-to-reservoir matrix
alph = .5; % learning rate
lamd = 0; % regularization parameter

Wout=RC_train(y_tr(:,1:end-1),xi_tr(:,2:end),A,Win,b,alph,lamd);

%% RC - predict
load RBC_data_test.mat % load testing data
y_te = S'*(X_te-meanX); % centered observations
y_te=y_te+.05*abs(y_te).*randn(size(y_te)); % add 5% noise
xi_pred=RC_pred(y_te(:,1:end-1),A,Win,b,alph,Wout); % compute kernel vector

%% S-DEIM estiamte
X_SDEIM = meanX+Phi*pinv(S'*Phi)*y_te(:,2:end)+Phi*Z*xi_pred;

%% Q-DEIM estimate
X_QDEIM=meanX+Phi*pinv(S'*Phi)*y_te(:,2:end);

%% Best fit
X_best = meanX+Phi*Phi'*(X_te(:,2:end)-meanX);

%% Computing the errors
err_best = vecnorm(X_te(:,2:end)-X_best)./vecnorm(X_te(:,2:end)-meanX);
err_SDEIM = vecnorm(X_te(:,2:end)-X_SDEIM)./vecnorm(X_te(:,2:end)-meanX);
err_QDEIM = vecnorm(X_te(:,2:end)-X_QDEIM)./vecnorm(X_te(:,2:end)-meanX);

%% plotting errors
plot(time_te(2:end),err_best*100,'r','LineWidth',1); hold on
plot(time_te(2:end),err_SDEIM*100,'b','LineWidth',2); hold on
plot(time_te(2:end),err_QDEIM*100,'k','LineWidth',1); hold on
set(gca,'fontsize',16)
xlabel('time','fontsize',24,'interpreter','latex')
ylabel('Relative Error (\%)','fontsize',24,'interpreter','latex')
legend('Best, 18\%, ','S-DEIM, 26\%','Q-DEIM, 68\%','Location','northeast','interpreter','latex')

%% plotting snapshots at t=100
figure
x=linspace(0,4,129);
y=linspace(0,1,33);
[x,y]=meshgrid(x,y);
subplot(2,2,1)
T_test = reshape(X_te(:,end)-meanX,33,129);
pcolor(x,y,T_test); shading interp; axis equal tight; colormap jet; set(gca,'fontsize',16)
title('Truth')
subplot(2,2,2)
T_best = reshape(X_best(:,end)-meanX,33,129);
pcolor(x,y,T_best); shading interp; axis equal tight; colormap jet; set(gca,'fontsize',16)
title('Best fit')
subplot(2,2,3)
T_QDEIM = reshape(X_QDEIM(:,end)-meanX,33,129);
pcolor(x,y,T_QDEIM); shading interp; axis equal tight; colormap jet; set(gca,'fontsize',16)
title('Q-DEIM')
subplot(2,2,4)
T_SDEIM = reshape(X_SDEIM(:,end)-meanX,33,129);
pcolor(x,y,T_SDEIM); shading interp; axis equal tight; colormap jet; set(gca,'fontsize',16)
title('S-DEIM')