function mout = available(matrix)
mout = matrix;
[m,n] = size(matrix);
for i = 1:m
    for j = 1:n
        if matrix(i,j)==2
            try
            mout = tryadd(mout,i+1,j);
            end
            try
            mout = tryadd(mout,i-1,j);
            try
            end
            mout = tryadd(mout,i,j+1);
            end
            try
            mout = tryadd(mout,i,j-1);
            end
%             try
%             mout = tryadd(mout,i+1,j+1);
%             end
%             try
%             mout = tryadd(mout,i-1,j-1);
%             end
%             try
%             mout = tryadd(mout,i-1,j+1);
%             end
%             try
%             mout = tryadd(mout,i+1,j-1);
%             end
        end
    end
end