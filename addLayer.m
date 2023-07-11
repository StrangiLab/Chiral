function [matrix,model,namecell,counter] = addLayer(matrix,prob,model,namecell,counter) 
[m,n] = size(matrix);
for i = 1:m
    for j = 1:n
        if matrix(i,j)==1 && rand()<prob                 
            try                
               
                model.geom('geom1').create(strcat('blkgb',num2str(counter)), 'Block');
                model.geom('geom1').feature(strcat('blkgb',num2str(counter))).set('size', {'wb' 'wb' 'hb'});
                model.geom('geom1').feature(strcat('blkgb',num2str(counter))).set('base', 'corner');
                model.geom('geom1').feature(strcat('blkgb',num2str(counter))).set('pos', [(getcellindex(mphgetexpressions(model.param),'wb')*(j-1))-getcellindex(mphgetexpressions(model.param),'blc')/2,...
                    (getcellindex(mphgetexpressions(model.param),'wb')*(i-1))-getcellindex(mphgetexpressions(model.param),'blc')/2, -getcellindex(mphgetexpressions(model.param),'hb')/2]);
                
                namecell{counter} = strcat('blkgb',num2str(counter));
                matrix(i,j) = 2;
                counter = counter + 1;
            catch
                fprintf('\nError in defining the added geometry: %2d,%2d\n',i,j)
                continue
            end
            try
                model.geom('geom1').runPre('fin');
            catch
                fprintf('\nNew Geometry did not compile.\n')
            end
                
        end
    end
end