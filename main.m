close all;
clear all;
clc;

% Read image file
% Image matrix (h,w) is regarded as data matrix (N,dim)
I = im2single(rgb2gray(imread('./lenna.png')));
% I = imresize(I,1/2);
U = repmat(mean(I,1),[size(I,1),1]);
B = I - U;

% PCA for B
covB = B'*B / (size(B,2)-1);
[V,D] = eig(covB); % covB*V = V*D, V:principal components (row vector), D:coefficients
[~,ind] = sort(diag(D),'descend');
V = V(:,ind); % sort principal components in descending order of coefficients

% Reconstruct image
ln = [1, 3, 5, 10, 50, size(B,2)]; % latent number for reducing data dimension
rI = cell(numel(ln),1);
for i = 1:1:numel(ln) 
    Z  = B*V(:,1:ln(i));  % Z:pricipal projections (score) (data dimension is reduced by ln(i))
    rB = Z*V(:,1:ln(i))'; % reconstruction from Z, B=B*V*V'=Z*V' (V*V'=I, V'=V^(-1), V is orthogonal matrix (íºçsçsóÒ))
    rI{i} = rB + U;
end

% Show results
figure;
subplot(3,3,1);
imshow(I); title('input image');
for i = 1:1:numel(ln)
    subplot(3,3,i+3);
    imshow(rI{i}); title(['PC:',num2str(ln(i))]);
end