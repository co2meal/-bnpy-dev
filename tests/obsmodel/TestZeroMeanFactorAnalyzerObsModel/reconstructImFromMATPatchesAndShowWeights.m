function [RIm, WIm] = reconstructImFromMATPatchesAndShowWeights( ...
                        matfilepath, imgID, patchWeightVec, ...
                        doShow, doSave, saveName)
%% reconstructImFromPatches
% Load patches from saved MAT file into original image and weight image.
%
% Allows showing custom values associated with each image patch. 
%% INPUT
% matpath : string filepath to the saved MAT file
% imgID : integer
% patchWeightVec : array with one entry per patch
% doShow : boolean indicator. If true, will display both images as figures.
%% OUTPUT
% RIm : reconstructed image, as a 2D array
% WIm : weight image, as a 2D array
%% EXAMPLE
% reconstructImFromMATPatchesAndShowWeights('Patches_Size8x8_Stride4/Batch001.mat', 1, linspace(1, 10000, 10000), 1);

Q = load(matfilepath);
if ~isfield(Q, 'stride')
    Q.stride = 4;
end
if ~isfield(Q, 'patchH')
    Q.patchH = 8;
    Q.patchW = 8;
end

nRowOff = length(Q.offsetVals);
nColOff = nRowOff;
nOff = nRowOff * nColOff;
nPatchPerImg_NoOverlap = Q.nPatchPerImg / nOff;

if isstr(imgID)
    imgIDstr = imgID;
    CellMatches = strfind(Q.BatchFileNames, imgIDstr);
    imgID = -1;
    for b = 1:Q.nImgPerBatch
       if ~isempty(CellMatches{b})
           imgID = b;
           break;
       end
    end
    if (imgID < 1)
       error(['Cannot find desired image: ' imgIDstr]); 
    end
else
    if (imgID < 1) || (imgID > Q.nImgPerBatch)
        error('Numeric imgID must be in range 1 <= imgID <= nImgPerBatch');
    end
end

% Grab relevant rows of big matrix X.
start = (imgID - 1) * Q.nPatchPerImg;
Xim = Q.X(start+1:start+nPatchPerImg_NoOverlap, :);
Xim = Q.X(start+1:start+Q.nPatchPerImg, :);

H = Q.BatchEffImgSizes(imgID, 1);
W = Q.BatchEffImgSizes(imgID, 2);
RIm = zeros(H, W);
WIm = zeros(H, W);
WIm = nan(H, W);

nRow = H/Q.patchH;
nCol = W/Q.patchW;

for patchID = 1:nRow*nCol
    % c is integer in 0, 1, ... nRow
    c = floor((patchID-1) / nRow);
    
    % c is integer in 0, 1, ... nRow
    r = mod(patchID, nRow);
    r = mod(r-1, nRow);
    
    % Rescale by the patch size
    r = r * Q.patchH;
    c = c * Q.patchW;
    RIm(r+1:r+Q.patchW, c+1:c+Q.patchW) = reshape(Xim(patchID, :), [Q.patchH, Q.patchW]);
    WIm(r+1:r+Q.patchW/2, c+1:c+Q.patchW/2) = patchWeightVec(patchID);
    WIm(r+1:r+Q.patchW/2, c+1+Q.patchW/2:c+Q.patchW) = patchWeightVec(patchID+nRow*nCol);
    WIm(r+1+Q.patchW/2:r+Q.patchW, c+1:c+Q.patchW/2) = patchWeightVec(patchID+2*nRow*nCol);
    WIm(r+1+Q.patchW/2:r+Q.patchW, c+1+Q.patchW/2:c+Q.patchW) = patchWeightVec(patchID+3*nRow*nCol);
end


if doShow
    figure(1);
    subplot(1, 2, 1); imagesc(RIm); colormap('gray'); axis('image');
    subplot(1, 2, 2); h=imagesc(WIm); axis('image');
    colormap('summer');
    set(h,'alphadata',~isnan(WIm));
%     threshold = 0.2 * (nanmax(patchWeightVec) - nanmin(patchWeightVec));
%     caxis([nanmin(patchWeightVec) - threshold,...
%            nanmax(patchWeightVec) + threshold]);
    set(gcf,'color',[1,1,1]);
    if doShow == 2
        figure(2);
        hist(patchWeightVec(~isnan(patchWeightVec)));
        set(gca, 'fontsize', 20);
        set(gcf,'color',[1,1,1]);
    end
    if doSave
        figure(1);
        export_fig([saveName,'.pdf']);
%         export_fig([saveName,'.pdf'], '-opengl');
        if doShow == 2
            figure(2);
            export_fig([saveName,'_hist.pdf']);
        end
    end
end
end