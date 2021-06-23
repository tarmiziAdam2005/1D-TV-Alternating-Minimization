% 23/6/2021,This is the core solver. 
% This is the script that implements the AM algorithm for our simple 1D TV
% Signal Denoising. 
%Study this code carefully and relate with all the equations in the slide
%!!!
function out = AM_1D(b, A, mu, rho, Nit, tol)

%Initializations

x         = b;          
N         = length(x);
funcVal   = zeros(Nit,1);
relError = zeros(Nit,1);


[D,DT,DTD] = defDDt(N); % In the slides we use nabla symbol for this (nabla is the upside down triangle symbol)
                        % the T here stands for the transpose.
                        % D = nabla, DT = nabla transpose, DTD = nabla
                        % transpose nabla. Plese relate to the equation of
                        % linear system in slides.


AT = A';     %AT = A trasposed
ATA = A'*A;  % ATA = A transposed A. See where we use this in slide tutorial.

for k = 1:Nit
    
    %%%%%% z-subproblem %%%%%%
    Dx = D*x;
    z =shrink(Dx,1/rho); % the "shrinkage" or "soft thresholding" operation.
                         % Refer to its definition at the end of this code.
    
    %%%%%% x-subproblem %%%%%%
    % Here is where we solve the linear system.
    
    lhs = ATA + (rho/mu)*DTD; % left hand side of our linear system. Can you relate this to the equation in slides ?
    rhs = AT*b + (1/mu)*DT*z; % Right hand side of the linear system. Relate this to the linear system equation in slides
    
    x_old = x; % store the previous iteration restored signal. To compute the tol
    
    [x,~] = cgs(lhs,rhs,1e-5,10); % AHA ! here we solve the linear system using the conjugate gradient method.
                                   % conjugate gradient method is build in
                                   % MATLAB. Please look into the help
                                   % documents on this cgs function !
    
    
    relError(k)    = norm(x - x_old,'fro')/norm(x, 'fro');
    funcVal(k)   = (mu/2)*norm(A*x - b)^2 + (rho/2)*norm(z - D*x)^2 +sum(abs(D*x));
    
    if relError(k) < tol % If error between current restored signal and previous restored signal is less than tol. Exit program !
        break;
    end
    
    
end


out.sol = x;
out.funVal = funcVal(1:k);
out.relativeError       = relError(1:k);
end



function [D,DT,DTD] = defDDt(N)
%Create a first order difference matrix D (Nabla)
% Please study this code to know what is actually nabla !
e = ones(N,1);
B = spdiags([e -e], [1 0], N, N);
B(N,1) =  1;

D = B; 
clear B;
% Create the transpose of D
DT = D'; %Remember that DT = -D, also called the backward difference.
DTD = D'*D;
end

function z = shrink(x,r)
z = sign(x).*max(abs(x)- r,0); % Soft thresholding operator. Similar to the equation in the slides right ?
end