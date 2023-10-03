clear
clc
%%
base_dir = 'F:\GAN_data\RealData2010';
sec_dir = 'gnf_images_MitoLyso1\images\MitoLyso';
store_base_dir = [base_dir,'\MLMask'];
train_dir = [base_dir,'\',sec_dir];
mkdir(store_base_dir);
Files = dir(train_dir);
TotalFluoThresh = 1.5e7;
for n=1:numel(Files)
    tmpFile = Files(n).name
    if numel(strfind(tmpFile,'.')) || numel(strfind(tmpFile,'..'))
        continue
    elseif numel(strfind(tmpFile,'README'))
        copyfile([train_dir,'\',tmpFile], store_base_dir)
    else
        store_img_dir = [store_base_dir, '\',tmpFile];
        mkdir(store_img_dir)
        copyfile([train_dir,'\',tmpFile,'\factors.mat'], store_img_dir)
        load([train_dir,'\',tmpFile,'\factors.mat'])
        for idx = 1:2:69
            tmpimg = sprintf('Stack-%05d.bmp',idx);
            tmpimgdna = sprintf('Stack-%05d.bmp',idx+1);
            image_ = imread([train_dir,'\',tmpFile,'\',tmpimg]);
            imagedna_ = imread([train_dir,'\',tmpFile,'\',tmpimgdna]);
            factor = factors(idx);
            factordna = factors(idx+1);
            image = double(image_)*factor;
            imagedna = double(imagedna_)*factordna;
            if sum(imagedna(:)) < TotalFluoThresh
                continue
            else
                [procimg,mask_img,~] = ml_preprocess(image,[],'ml','yesbgsub');
                [procdna,mask,res] = ml_preprocess(imagedna,[],'ml','yesbgsub');
                [dnas,dnanum] = bwlabel(mask);
                if dnanum <= 4
                    impurity = imdilate(dnas>0,strel('disk',15));
                    imagedna(impurity) = 0;
                    [procdna,mask,res] = ml_preprocess(imagedna,[],'ml','yesbgsub');
                end
                tmpimgmask = sprintf('StackMask-%05d.bmp',idx);
                tmpimgdnamask = sprintf('StackMask-%05d.bmp',idx+1);
                imwrite(mask_img,[store_img_dir,'\',tmpimgmask])
                imwrite(mask,[store_img_dir,'\',tmpimgdnamask])
                clear mask_img mask procimg procdna
            end
        end
    end
end


