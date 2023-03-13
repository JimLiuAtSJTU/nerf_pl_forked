
clear all
close all

u=0;
std=0.1;


x=-std*5:0.01:std*5;

y= 1./sqrt(2*pi)/std.*exp(-(x-u).^2/2/std.^2);

y=y*length(x);
plot(x,y)
