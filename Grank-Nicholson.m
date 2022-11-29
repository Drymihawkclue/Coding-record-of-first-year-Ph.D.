clear all;
m=35
n=50
u4=zeros(m,n);
DT=0.1;
A=1;
H=2*pi/(n-2)
for j=1:n
   u4(1,j)=sin(H*(j-1))
end
RD=DT/H;
B=u4;
delta=1.0;
while delta>=0.01;
  u4(2:m,2:n-1)=B(1:m-1,2:n-1)-A*RD/4*(B(1:m-1,3:n)-B(1:m-1,1:n-2)+B(2:m,3:n)-B(2:m,1:n-2));%Grank-Nicholson∏Ò Ω
  u4(2:m,1)=u4(2:m,n-1)
  u4(2:m,n)=u4(2:m,2) 
  delta=max(max((abs(u4-B)./(u4+eps))))
  B=u4;
end
mesh(u4)