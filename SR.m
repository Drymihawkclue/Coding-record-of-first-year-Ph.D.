function [x]=SR(x,a,b,n,w)
for i=1:n-1
    maximum=abs(a(i,1));
    maxIndex=i;
    for j=2:n
        if abs(a(i,j))>maximum
            maximum=a(i,j);
            maxIndex=j;
        end
    end
    temp=a(maxIndex,:);
    a(maxIndex,:)=a(i,:);
    a(i,:)=temp
    bt=b(maxIndex);
    b(maxIndex)=b(i);
    b(i)=bt
end
for v=1:n
    b(v)=b(v)/a(v,v);
end
for u=1:n
    a(u,1:n)=-a(u,1:n)/a(u,u);
end
a(logical(eye(size(a))))=0;
delta=10.0;
m=0;
while delta>0
    t=x;
  for h=1:n
    t(h)=(1-w)*t(h)+w*(a(h,:)*t'+b(h));
  end
  d=t-x;
  delta=max(abs(d));
  x=t;
  m=m+1
end
end