function Wout=RC_train(Xin,Xout,A,Win,b,alph,lamd)

k=size(Xin,2); % number of snapshots
m=size(A,1); % reservoir dim

R=nan(m,k); % reservoir matrix
r = randn(m,1); % initialize reservoir state

for i = 1:k
    r = (1-alph)*r+alph*tanh(A*r+Win*Xin(:,i)+b); % update resevoir state
    R(:,i) = r; % update reservoir matrix
end

Wout=((R*R'+lamd*eye(m))\(R*Xout'))'; % output matrix

end