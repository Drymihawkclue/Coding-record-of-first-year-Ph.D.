clear all;
m=50
n=42
u2=zeros(m,n);
DT=0.1;
A=1;
H=2*pi/(n-2)
for j=1:n
   u2(1,j)=sin(H*(j-1))
end
RD=DT/H
for i=1:m-1
    for j=2:n-1
        u2(i+1,j)=RD/2*(u2(i,j-1)-u2(i,j+1))+(u2(i,j+1)+u2(i,j-1))/2;%lax-friderichs∏Ò Ω
    end
    u2(i+1,1)=u2(i+1,n-1)
    u2(i+1,n)=u2(i+1,2) 
end
mesh(u2)
