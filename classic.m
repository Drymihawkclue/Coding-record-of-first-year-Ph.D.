clear all;
m=35
n=50
u3=zeros(m,n);
DT=0.1;
A=1;
H=2*pi/(n-2)
for j=1:n
   u3(1,j)=sin(H*(j-1))
end
RD=DT/H;
B=u3;
delta=1.0;
while delta>=0.01;
  u3(2:m,2:n-1)=B(1:m-1,2:n-1)-A*RD/2*(B(2:m,3:n)-B(2:m,1:n-2));%π≈µ‰∏Ò Ω
  u3(2:m,1)=u3(2:m,n-1)
  u3(2:m,n)=u3(2:m,2) 
  delta=max(max((abs(u3-B)./(u3+eps))))
  B=u3;
end
mesh(u3)