function K = Kappa(X)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
n = length(X);
K = 0;

for i = 1:n
    for j = 1:n
        K = K + norm(X(i,:)-X(j,:))^2;
    end
end

K = K/(n^2);


end

