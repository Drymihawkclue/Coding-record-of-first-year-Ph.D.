function [x]=Gauss(a,b,n)
for i=1:n-1
    max=abs(a(i,i));
    maxIndex=i;
    for j=i+1:n
        if abs(a(j,i))>max
            max=a(j,i);
            maxIndex=j;
        end
    end
    temp=a(maxIndex,:);
    a(maxIndex,:)=a(i,:);
    a(i,:)=temp;
    bt=b(maxIndex);
    b(maxIndex)=b(i);
    b(i)=bt;
    Det=det(a(1:i,1:i));
    if (Det==0)
        error('This matrix cannot be solved by Gauss algorithm');
    end
    %消元计算
    for k=i+1:n
        h=a(k,i)/a(i,i);
        b(k)=b(k)-h*b(i);
        for u=i+1:n
            a(k,u)=a(k,u)-h*a(i,u);
        end
        disp(a);
        disp(b);
    end
end
%回代
x(n)=b(n)/a(n,n);
for m=n-1:-1:1
    x(m)=(b(m)-(a(m,m+1:n)*x(m+1:n)'))/a(m,m);
end
end