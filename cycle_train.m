close all;clear all;clc;
w=32;
pathlist1={'PRPS_Diffuse','PRPS','PRPS'};
pathlist2={'apple', 'Ball', 'buddha', 'bunnydoll', 'ear', 'eden','kitty', 'littledoll', 'mother', 'pig','statue2','stone','uncle','versace', 'wolf'};
pathlist3={'images_diffuse','images_specular','images_metallic'};
mkdir('../PRPS_Dataset');
mkdir('../PRPS_Dataset/Images');

for s = 1:15
    mkdir(['../PRPS_Dataset/Images/' pathlist2{s}]);
end
count = 1;

fid1 = fopen('../PRPS_Dataset/Images/mtrl.txt' ,'wt');
for p = 1 : 3
    path1=pathlist1{1,p};
    path3=pathlist3{1,p};
    if p==1
        scale=2;
    elseif p==2||p==3
        scale=4;
    end
    for s = 1 : 15
        path2=pathlist2{1,s};
        mkdir(['../PRPS_Dataset/Images/' pathlist2{s} '/' pathlist3{p}]);
        fprintf(fid1,'%s\n',[pathlist2{s} '/' pathlist3{p}]);
        load(['./' path1 '/' path2 '.obj/light_mat.']);
        L=L';
%         Lmask=L(:,3)<sin(20/180*pi);
% 
%         Lind=find(Lmask==0);
%         L=L(Lind,:);
% 
%         x= 0.5*(L(:,1)+1)*(w-1); 
%         x=uint32(x);
%         y= 0.5*(L(:,2)+1)*(w-1);
%         y=uint32(y);
%         mapind=y*w+x+1;
        Normal_gt=imread(['./' path1 '/' path2 '.obj/gt_normal.tif']);
        [row,line,z]=size(Normal_gt);
        if p == 1
            Normal_gt = Normal_gt(1:scale:row,1:scale:line,:);
            Normal_gt = uint8(double(Normal_gt)/65535 * 255);
            mask =  sum(Normal_gt,3) == 0;
            mask = repmat(mask,[1,1,3]);
            Normal_gt(mask) = 127;
            imwrite(Normal_gt,['../PRPS_Dataset/Images/' pathlist2{s} '/' pathlist2{s} '_normal.png' ]);
        end
%         Normal_gt=im2double(Normal_gt);
%         Normal_gt=Normal_gt.*2-1;
%         maskin=imread(['./' path1 '/' path2 '.obj/inboundary.png']);
%         maskin = maskin(1:scale:row,1:scale:line,:);
%         maskon=imread(['./' path1 '/' path2 '.obj/onboundary.png']);
%         maskon = maskon(1:scale:row,1:scale:line,:);
%         maskindin=find(maskin==1);
%         maskindon=find(maskon==1);
%         maskind=union(maskindin,maskindon);
%         normals=reshape(Normal_gt,[size(Normal_gt,1)*size(Normal_gt,2) 3]);
%         normals=normals(maskind,:); 
        
%         normals_gt = zeros(size(Normal_gt, 1) * size(Normal_gt, 2), 3);
%         normals_gt(maskind, :) = normals;
%         normals_gt = reshape(normals_gt, size(Normal_gt, 1), size(Normal_gt, 2), 3);
        
        fid = fopen(['../PRPS_Dataset/Images/' pathlist2{s} '/' pathlist3{p} '/' pathlist2{s} '_' pathlist3{p} '.txt'],'wt');
        for i=1:size(L,1)
%             img=imread(['./' path1 '/' path2 '.obj/' path3 '/' num2str(Lind(i)-1,'%05d') '.tif']);    
            img=imread(['./' path1 '/' path2 '.obj/' path3 '/' num2str(i-1,'%05d') '.tif']); 
            img = img(1:scale:row,1:scale:line,:);
            img = uint8((double(img)/65535 * 255));
            imwrite(img,['../PRPS_Dataset/Images/' pathlist2{s} '/' pathlist3{p} '/l_' num2str(i - 1,'%05d') ',' ...
                num2str(L(i,1),'%.2f') ',' num2str(L(i,2),'%.2f') ',' num2str(L(i,3),'%.2f') '.png']);
            fprintf(fid, '%s', ['l_' num2str(i - 1,'%05d') ',' ...
                num2str(L(i,1),'%.2f') ',' num2str(L(i,2),'%.2f') ',' num2str(L(i,3),'%.2f') '.png']);
            fprintf(fid, ' %.4f', L(i,1));
            fprintf(fid, ' %.4f', L(i,2));
            fprintf(fid, ' %.4f\n', L(i,3));
%             img=im2double(img);
%             img=rgb2gray(img);
        end
        fclose(fid); 
        disp([p,s]);
    end
end
fclose(fid1);