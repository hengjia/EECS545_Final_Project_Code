clc;
clear all;

% Load data
load hand_written_digits_545;
training_label(1:5000)=10*ones(5000,1);

% Suppose we have l labeled examples and u unlabeled examples. We will
% train these mixed data with semi-supervised learning. For each set, the
% first (l/10) data are labeled and the last (u/10) data are unlabeled.

l = 10; % Number of labeled examples
u = 990; % Number of unlabeled examples
t = 50; % The constant for heat kernel weights
training_d = [];
data_label = [];

% First labeled data
% for i=2:3
%     training_d = [training_d;training_data((i-1)*5000+1:(i-1)*5000+(l/2),:)];
%     data_label = [data_label;training_label((i-1)*5000+1:(i-1)*5000+(l/2))];
% end

% Set 4 and 9 as the training data
training_d = [training_d;training_data(4*5000+1:4*5000+(l/2),:)];
training_d = [training_d;training_data(9*5000+1:9*5000+(l/2),:)];
data_label = [data_label;ones(l/2,1)];
data_label = [data_label;-ones(l/2,1)];

% Then unlabeled data
% for i=2:3
%     training_d = [training_d;training_data(i*5000-(u/2)+1:i*5000,:)];
% end

% Set 4 and 9 as the training data
training_d = [training_d;training_data(5*5000-(u/2)+1:5*5000,:)];
training_d = [training_d;training_data(10*5000-(u/2)+1:10*5000,:)];

% Step 1: Calculate the graph kernel with heat kernel weight
W = zeros(l+u,l+u); 
for i=1:(l+u)
    for j=1:(l+u)
        W(i,j)=exp(-norm(training_d(i,:)-training_d(j,:))^2/(4*t));
    end
end

% Step 2: Calculate the gram matrix with homogeneous polynomial kernel with
% degree 1
K = zeros(l+u,l+u);
for i=1:(l+u)
    for j=1:(l+u)
        K(i,j)=training_d(i,:)*training_d(j,:)';
    end
end

% Step 3: Calculate graph Laplacian matrix: L = D-W
D = zeros(l+u,l+u);
for i=1:l+u
    D(i,i)=sum(W(i,:));
end
L = D-W;

% Step 4: Determine gamma_A and gamma_I
gamma_A = 0.025;
gamma_I = 0.05;

% Step 5: Calculate alpha
J = diag([ones(l,1);zeros(u,1)]);
Y = [data_label;zeros(u,1)];
alpha = (J*K+gamma_A*l*eye(l+u)+gamma_I*l/(u+l)^2*L*K)^(-1)*Y;

test_ans = alpha'*(training_d*training_d');
real_p = test_ans;
for i=1:l+u
    if (abs(test_ans(i)-1)<abs(test_ans(i)+1))
        test_ans(i) = 1;
    else
        test_ans(i) = -1;
    end
end

% Error rate for label data
ER_labeled = sum((test_ans(1:l)-data_label')~=0)/l;
ER_unlabeled = sum((test_ans(l+1:l+u)-[ones(u/2,1);-ones(u/2,1)]')~=0)/u;
ER_overall = (sum((test_ans(1:l)-data_label')~=0)+sum((test_ans(l+1:l+u)-[ones(u/2,1);-ones(u/2,1)]')~=0))/(l+u);

% Kmeans
k_means = kmeans(training_d,2);
ER_kmeans = (sum((k_means' - [ones(1,5) 2*ones(1,5) ones(1,495) 2*ones(1,495)])~=0))/(l+u);
