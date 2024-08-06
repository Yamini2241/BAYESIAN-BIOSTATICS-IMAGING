cloall; cleall; clc;
inpt_image = imread('kavya.jpg');
% CIR=imnoise(CIR,'poisson');
figure, imshow(inpt_image);title('pre processing of original image')
imwrite(inpt_image,'input.jpg'); title('CIR Composite (Un- enhanced)') 
H=fspecial('motion',10,45); motion_img=imfilter(inpt_image,H,'replicate')
; imshow(motion_img);title('Raw image'); % img = rgb2gray(motion_img);
text(size(inpt_image,2), size(inpt_image,1) + 15,... 'Image courtesy of Space Imaging, LLC',...
'FontSize', 7, 'HorizontalAlignment', 'right');process_image = decorrstretch(inpt_image,
'Tol',0.01); imwrite(process_image,'decorrCIR.jpg');
red_components = im2single(inpt_image(:,:,1)); % color separation R components
blue_components = im2single(inpt_image(:,:,3)); % color separation G components figure,
imshow(blue_components)
title('Visible maxima points'); figure, imshow(red_components) title('Near minima points');
concantation = (red_components - blue_components) ./(red_components + blue_components);
figure, imshow(concantation,'DisplayRange',[-1, 1])
title('Normalized pixels');threshold = 0.4; q = (concantation > threshold);
100 * numel(red_components(q(:))) /
numel(red_components); figure,imshow(q)
title('output with Threshold Applied');
%%%%%apply morphlogicaloperations morphological=strel('disk',1);
morpho=imerode(blue_components,morphological);%%adding the pixels to output image 
figure,imshow(mat2gray(morpho)); title('DETECT potential morphological operation');
%%%% appliying thresholding [rowscolumns]=size(morpho);
si=1; for i = 1:rows for j
=
1:columns con=0; s1=1;for k1
=isi:i+si for p1
= j-si:j+si
if ((k1 > 0 && p1 > 0) && (k1 < rows && p1 < columns
)) con = con+1; s1=s1*morpho(k1,p1); end out_loop(i,j)=s1^(1/con
); endend
figure,imshow(out_loop),title('threshold applying');out_loop =
double(imread('decorrCIR.jpg'))/255; % th = double(imread('decorrCIR.jpg'))/255; out_loop =
out_loop+0.01*rand(size(out_loop)); out_loop(out_loop<0) = 0; out_loop(out_loop>1)
= 1;
% Set Weight filter parameters. w = 5; % matrix width sigma = [3 0.1]; % Weight filter 
standard deviations
satillite_deforestation = optimal_coherent_processing_interval_selection(out_loop,w,sigma);
% optimal_coherent_processing_interval_selection
% Display color input image and filtered output.out_loop=imresize(out_loop,[1024
1024]);
satillite_deforestation=imresize(satillite_deforestation,[10241024]); figure, 
imshow(satillite_deforestation);
axis image; title('processing image');drawnow;
%%disp(' existing values') psnrv1=psnr(inpt_image,motion_img);
mse1=mse(inpt_image,motion_img)
; fprintf('PSNR
%f\n',psnrv1)fprintf('MSE %f\n',mse1)
%disp(' proposed values')
%psnrv2=psnr(inpt_image,satillite_deforestation);
%mse2=mse(inpt_image,[],satillite_deforestation,[]);
%fprintf('PSNR %f\n',psnrv2)
%fprintf('MSE %f\n',mse2)
% deforestation_image=rgb2hsv(satillite_deforestation);
