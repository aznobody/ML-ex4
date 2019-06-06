function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
        %X=5000x400
        %Theta1=25x401
        %Theta2= 10x26
        for i=1:m
            x=X(i,:)'; %'take each example 400x1
            x=[1;x];           %401x1
            z2=Theta1*x;        %25x1
            a2=sigmoid(z2);    %25x1
            a2=[1;a2];         %26x1  
            z3= Theta2*a2;     %10x1
            a3=sigmoid(z3);    %10x1
            h=a3; %hypothesis h(x)

            y_actual=zeros(num_labels,1); %10x1
            labelY=y(i); %here label '10' means '0', so if y(10)=1 then it means its '0';
            y_actual(labelY)=1;

            costOfExample=0;    
            costVector= -[y_actual.*log(h)+(1-y_actual).*log(1-h)]; %10x1
            costOfExample=sum(costVector);

            J+=costOfExample;
        end
        J=J/m;
        
        %Adding regularization
        R=0;
        for i=1:size(Theta1,1)  %25x401
            for j=2:size(Theta1,2) %j=1 is the bias term here not j=401
                R+=Theta1(i,j).^2;
            end
        end

        for i=1:size(Theta2,1)     %10x26
            for j=2:size(Theta2,2)  %j=1 is the bias term not j=26
                R+=Theta2(i,j).^2;
            end
        end
        J+=R*lambda/(2*m);

    


   

%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

        %X=5000x400
        %Theta1=25x401
        %Theta2= 10x26

        Theta2_unbiased=Theta2(:,2:size(Theta2,2)); %10x25
        Theta1_unbiased=Theta1(:,2:size(Theta1,2)); %25x400

        Del1=zeros(size(Theta1));
        Del2=zeros(size(Theta2));


        for i=1:m
            x=X(i,:)'; %'take each example 400x1
            x=[1;x];           % add bias term 401x1
            a1=x;
            z2=Theta1*a1;        %25x1
            a2=sigmoid(z2);    %25x1
            a2=[1;a2];         %26x1  
            z3= Theta2*a2;     %10x1
            a3=sigmoid(z3);    %10x1
            h=a3; %hypothesis h(x)

            y_actual=zeros(num_labels,1); %10x1
            labelY=y(i); %here label '10' means '0', so if y(10)=1 then it means its '0';
            y_actual(labelY)=1;

            %BackPropagate
            %Theta1 =25x401
            %Theta2= 10x26
            %a1=401x1
            %a2=26x1
            del3=a3-y_actual; %10x1          
            del2= (Theta2')*del3.*a2.*(1-a2); %' 26x1 (z2=26x1)
            del2=del2(2:end); %25x1
            

            %Del2=10x26
            %del3= 10x1
            %a2=26x1
            Del2 = Del2 + del3 * a2'; %10x26

            %Del1=25x401
            %del2= 25x1
            %a1=401x1
            Del1 = Del1 + del2 * a1'; %' 25x401    
            
        end
        Theta1_grad=Del1/m;
        Theta2_grad=Del2/m;

%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
%Theta1 =25x401=Theta1_grad
%Theta2= 10x26 =Theta2_grad
%Ignoring the first column
Theta1_grad(:,2:end)+= lambda/m* Theta1(:,2:end); 
Theta2_grad(:,2:end)+= lambda/m* Theta2(:,2:end);






% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
