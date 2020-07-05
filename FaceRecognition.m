function FaceRecognition
clear;  close all; clc;
allsamples=[];  %所有训练图像 
for i=1:40    
    for j=1:5        
        if(i<10)
           a=imread(strcat('E:\matlabDEMO\personFace\3975339\ORL\00',num2str(i),'0',num2str(j),'.bmp'));     
        else
            a=imread(strcat('E:\matlabDEMO\personFace\3975339\ORL\0',num2str(i),'0',num2str(j),'.bmp'));  
        end          
        b=a(1:112*92); % b是行矢量 1×N，其中N＝10304，提取顺序是先列后行，即从上到下，从左到右        
        b=double(b);        
        allsamples=[allsamples; b];  % allsamples 是一个M * N 矩阵，allsamples 中每一行数据代表一张图片，其中M＝200   
    end
end
samplemean=mean(allsamples); % 平均图片，1 × N  
imshow(mat2gray(reshape(samplemean,112,92)));  %对m个向量取平均值得平均图
title('平均图像');

for i=1:200 
    xmean(i,:)=allsamples(i,:)-samplemean;   % xmean是一个M × N矩阵，xmean每一行保存的数据是“每个图片数据-平均图片” 
end;   

sigma=xmean*xmean';     % M * M 阶矩阵 
[v,d]=eig(sigma);
d1=diag(d); 
[d2,index]=sort(d1);    %以升序排序 
cols=size(v,2);         %特征向量矩阵的列数

for i=1:cols      
    vsort(:,i) = v(:, index(cols-i+1) ); % vsort是一个M*col(注:col保存的是按降序排列的特征向量,每一列构成一个特征向量)   
    dsort(i)   = d1( index(cols-i+1) );  % dsort保存的是按降序排列的特征值，是一维行向量 
end  %完成降序排列

%以下选择90%的能量数 
dsum = sum(dsort);     
dsum_extract = 0;   
p = 0;     
while( dsum_extract/dsum < 0.9)       
    p = p + 1;          
    dsum_extract = sum(dsort(1:p));     
end

a=1:1:200;
for i=1:1:200
y(i)=sum(dsort(a(1:i)) );
end
figure
y1=ones(1,200);
plot(a,y/dsum,a,y1*0.9,'linewidth',2);
grid
title('前n个特征特占总的能量百分比');
xlabel('前n个特征值');
ylabel('占百分比');
figure
plot(a,dsort/dsum,'linewidth',2);
grid
title('第n个特征特占总的能量百分比');
xlabel('第n个特征值');
ylabel('占百分比');
i=1;  % (训练阶段)计算特征脸形成的坐标系
while (i<=p && dsort(i)>0)      
    base(:,i) = dsort(i)^(-1/2) * xmean' * vsort(:,i);   % base是N×p阶矩阵，除以dsort(i)^(1/2)是对人脸图像的标准化，特征脸
      i = i + 1; 
end

% 将训练样本对坐标系上进行投影,得到一个 M*p 阶矩阵allcoor
allcoor = allsamples * base; accu = 0;   % 测试过程
for i=1:40     
    for j=6:10 %读入40 x 5幅测试图像         
         if(i<10)
            if(j<10)
             a=imread(strcat('E:\matlabDEMO\personFace\3975339\ORL\00',num2str(i),'0',num2str(j),'.bmp'));     
            else
             a=imread(strcat('E:\matlabDEMO\personFace\3975339\ORL\00',num2str(i),num2str(j),'.bmp'));     
            end
         else
             if(j<10)
             a=imread(strcat('E:\matlabDEMO\personFace\3975339\ORL\0',num2str(i),'0',num2str(j),'.bmp'));     
            else
             a=imread(strcat('E:\matlabDEMO\personFace\3975339\ORL\0',num2str(i),num2str(j),'.bmp'));     
             end
        end      
        b=a(1:10304);        
        b=double(b);        
        tcoor= b * base; %计算坐标，是1×p阶矩阵      
        for k=1:200                 
            mdist(k)=norm(tcoor-allcoor(k,:));        
        end;             %三阶近邻   
        [dist,index2]=sort(mdist);          
        class1=floor((index2(1)-1)/5)+1;  %最大程度缩小差距
        class2=floor(index2(2)/5)+1;        
        class3=floor(index2(3)/5)+1;        
        if class1~=class2 && class2~=class3 
            class=class1;         
        elseif class1==class2          
            class=class1;         
        elseif class2==class3     
            class=class2;         
        end;         
        if class==i      
            accu=accu+1;        
        end;   
    end;
end;  
accuracy = accu/200     %输出识别率,200张图片
system('pause');
warndlg(['图像识别率：' strcat(num2str(accuracy*100),'%')], 'Accuracy','modal');

