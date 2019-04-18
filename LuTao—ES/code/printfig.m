clear;clc;
% 自变量
x = -5:0.05:5;  
y =  -5:0.05:5;  

% 因变量
xlen = length(x);
ylen = length(y);
z = zeros(ylen,xlen);
for i = 1:xlen
    for j = 1:ylen
            z(j,i) = x(i)*x(i)+y(j)*y(j)+20- 10*(cos(2*pi*x(i))+cos(2*pi*y(j)));

%         z(j,i) = sin(x(i)) + sin(y(j)); %反比例函数
    end
end

[xx,yy]=meshgrid(x,y);
surf(xx,yy,z);
shading interp


