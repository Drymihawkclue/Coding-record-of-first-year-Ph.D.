function [X]=TRILINE(A,B,C,D,N)
r(1)=C(1)/B(1);
y(1)=D(1)/B(1);
for k=2:N
    r(k)=C(k)/(B(k)-r(k-1)*A(k));
    y(k)=(D(k)-y(k-1)*A(k))/(B(k)-r(k-1)*A(k));
end
X(N)=y(N);
for k=N-1:-1:1
    X(k)=y(k)-r(k)*X(k+1);
end