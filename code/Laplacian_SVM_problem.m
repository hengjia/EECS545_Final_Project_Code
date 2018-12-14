clc;
clear;

% Load data
load hand_written_digits_545;
training_label(1:5000)=10*ones(5000,1);

% Suppose we have l labeled examples and u unlabeled examples. We will
% train these mixed data with semi-supervised learning. For each set, the
% first (l/10) data are labeled and the last (u/10) data are unlabeled.

l = 200; % Number of labeled examples
u = 1000-l; % Number of unlabeled examples
training_d = [];
data_label = [];

% Set 4 and 9 as the training data
training_d = [training_d;training_data(4*5000+1:4*5000+(l/2),:)];
training_d = [training_d;training_data(9*5000+1:9*5000+(l/2),:)];
data_label = [data_label;ones(l/2,1)];
data_label = [data_label;-ones(l/2,1)];

% Set 4 and 9 as the training data
training_d = [training_d;training_data(5*5000-(u/2)+1:5*5000,:)];
training_d = [training_d;training_data(10*5000-(u/2)+1:10*5000,:)];

% Step 1: Calculate the graph kernel with heat kernel weight
options.GraphWeights = 'heat';
options.GraphWeightParam = 50;
options.NN = 1;
options.GraphDistanceFunction = 'euclidean';
W = adjacency(options,training_d);


% Step 2: Calculate the gram matrix with homogeneous polynomial kernel with
% degree 1
K = zeros(l+u,l+u);
for i=1:(l+u)
    for j=1:(l+u)
        K(i,j)=training_d(i,:)*training_d(j,:)';
    end
end
% options.Kernel = 'poly';
% options.KernelParam = 1;
% K=calckernel(options,training_d,training_d);

% Step 3: Calculate graph Laplacian matrix: L = D-W
D = zeros(l+u,l+u);
for i=1:l+u
    D(i,i)=sum(W(i,:));
end
L = D-W;

% Step 4: Determine gamma_A and gamma_I
gamma_A = 0.05;%1
gamma_I = 0.025;%1e-5

%step 5*
% generating default options


% initial alpha vector and bias
alpha=zeros(u+l,1); 
Y = [data_label;zeros(u,1)];
alpha = newton(gamma_A,gamma_I,Y,L,K,alpha,l,u+l);

% Step 5: Calculate alpha
J = diag([ones(l,1);zeros(u,1)]);


svs=find(alpha~=0);
test_ans=sign(K(:,svs)*alpha);
test_ans = test_ans';

% Error rate for label data
ER_labeled = sum((test_ans(1:l)-data_label')~=0)/l;
ER_unlabeled = sum((test_ans(l+1:l+u)-[ones(u/2,1);-ones(u/2,1)]')~=0)/u;
ER_overall = (sum((test_ans(1:l)-data_label')~=0)+sum((test_ans(l+1:l+u)-[ones(u/2,1);-ones(u/2,1)]')~=0))/(l+u);


%{
@article{melacci2011primallapsvm,
  title={{Laplacian Support Vector Machines Trained in the Primal}},
  author={Melacci, Stefano and Belkin, Mikhail},
  journal={Journal of Machine Learning Research},
}
%}

function alpha = newton(gamma_A,gamma_I,Y,L,K,alpha,l,n)
% {newton} trains the classifier using the Newton's method.

    labeled=Y~=0;
    t=0;
    sv=false(n,1);
    if gamma_I~=0, LK=L*K; end

    while 1  

    sv_prev=sv; 
    sv=labeled;
     
    % goal conditions
    if t>=200, break, end           
    if isequal(sv_prev,sv), break, end 
    t=t+1;

    IsvK=sparse([],[],[],n,n,l*n);

    if gamma_I==0 % SVM (sparse solution)
        alpha_new=zeros(n,1);
        alpha_new(sv)=(gamma_A*speye(l)+IsvK(sv,sv))\Y(sv);
    else % LapSVM
        IsvY=sparse([],[],[],n,1,l); 
        IsvY(sv)=Y(sv);             
        alpha_new=(gamma_A*speye(n)+IsvK+gamma_I*LK)\IsvY;
    end

    % step
    alpha=alpha_new;
    end
end

