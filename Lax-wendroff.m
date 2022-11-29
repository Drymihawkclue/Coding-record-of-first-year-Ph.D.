clear all;
m=50
n=42
u1=zeros(m,n);
DT=0.1;
A=1;
H=2*pi/(n-2)
for j=1:n
    u1(1,j)=sin(H*(j-1))
end
RD=DT/H
for i=1:m-1
    for j=2:n-1
        u1(i+1,j)=u1(i,j)+A*RD/2*(u1(i,j+1)-u1(i,j-1))+A*A*RD*RD/2*(u1(i,j+1)+u1(i,j-1)-2*u1(i,j))%lax-wendroff∏Ò Ω
    end
    u1(i+1,1)=u1(i+1,n-1)
    u1(i+1,n)=u1(i+1,2)    
end
mesh(u1)

