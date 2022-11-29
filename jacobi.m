function [x]=jacobi(x,a,b,n)
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
%     if n>2
%        t(1)=(b(1)-a(1,2:n)*x(2:n)')/a(1,1);
%        t(n)=(b(n)-a(n,1:n-1)*x(1:n-1)')/a(n,n);
%        for i=2:n-1
%             t(i)=(b(i)-a(i,i+1:n)*x(i+1:n)'-a(i,1:i-1)*x(1:i-1)')/a(i,i);
%        end
%     end
%     if n==2
%        t(1)=(b(1)-a(1,2:n)*x(2:n)')/a(1,1);
%        t(n)=(b(n)-a(n,1:n-1)*x(1:n-1)')/a(n,n);
%     end
for v=1:n
    b(v)=b(v)/a(v,v);
end
for u=1:n
    a(u,1:n)=-a(u,1:n)/a(u,u);
end
    a(logical(eye(size(a))))=0
    delta =1.0;
    m=0;
while delta>0.0  
    t=a*x'+b';
    d=t-x';
    delta=max(abs(d));
    x=t';
    m=m+1
end
    
end

    