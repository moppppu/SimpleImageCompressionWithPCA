clear all;
close all;
clc;

%% Simple Image Compression With PCA

% Read image file
% Image matrix (h,w) is regarded as data matrix (N,dim)
I = im2single(rgb2gray(imread('./lenna.png')));
U = repmat(mean(I,1),[size(I,1),1]); % average along with N (not dim)
X = I - U;

% PCA for X
covX = X'*X / (size(X,2)-1);
[V,D] = eig(covX); % covX*V = V*D, V:principal components (row vector), D:coefficients
[~,ind] = sort(diag(D),'descend');
V = V(:,ind); % sort principal components in descending order of coefficients

% Reconstruct image
ln = [1, 5, 10, 25, 50, size(X,2)]; % latent number for reducing data dimension
rI = cell(numel(ln),1);
for i = 1:1:numel(ln) 
    Z  = X*V(:,1:ln(i));  % Z:pricipal projections (score) (data dimension is reduced by ln(i))
    rX = Z*V(:,1:ln(i))'; % reconstruction from Z, X=X*V*V'=Z*V' (V*V'=I, V'=V^(-1), V is orthogonal matrix (íºçsçsóÒ))
    rI{i} = rX + U;
end

% Show reconstruct image
figure;
subplot(3,3,1);
imshow(I); title(['input image *(h,w)=(',num2str(size(I,1)),',',num2str(size(I,2)),')']);
for i = 1:1:numel(ln)
    subplot(3,3,i+3);
    imshow(rI{i}); title(['Using PC:1-',num2str(ln(i))]);
end