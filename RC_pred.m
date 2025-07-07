function Xpred=RC_pred(Xin,A,Win,b,alph,Wout)

k=size(Xin,2); % number of snapshots
n=size(Wout,1); % output dimension
m=size(A,1); % reservoir dimension

r = zeros(m,1); % initialize reservoir state
Xpred=nan(n,k); % placeholder for outputs

for i = 1:k
    r = (1-alph)*r+alph*tanh(A*r+Win*Xin(:,i)+b); % update reservoir state
    Xpred(:,i) = Wout*r; % compute output
end


end