counter=92;
for n1 = 1:6
    for n2 = 1:7
        try
            file = strcat('T:/Users/Alpha/Documents/Andy/chiral_data_1of2/',num2str(n1),'/basemodel',num2str(n2),'.mph');
            model = mphload(file);
        catch
            fprintf('\n Set %d: elements %d; coutner %d\n',n1,n2-1,counter)
            break
        end
        try
            heights = {'hb','-hb','2.5*hb','-2.5*hb','5*hb','-5*hb'};
            for i = 1:6
                data = strcat('data',num2str(i+100));
                model.result.export.create(data, 'Data');
                model.result.export(data).set('data', 'dset2');
                model.result.export(data).setIndex('looplevelinput', 'manualindices', 1);
                model.result.export(data).setIndex('looplevelindices', 201, 1);
                model.result.export(data).setIndex('expr', 'C_CPL/C0', 0);
                model.result.export(data).set('exporttype', 'vtu');
                model.result.export(data).set('location', 'grid');
                model.result.export(data).set('gridstruct', 'grid');
                model.result.export(data).set('gridx3', 'range(-blc/2,(blc/2-(-blc/2))/49,blc/2)');
                model.result.export(data).set('gridy3', 'range(-blc/2,(blc/2-(-blc/2))/49,blc/2)');
                model.result.export(data).set('gridz3', heights{i});
                model.result.export(data).set('exporttype', 'text');
                filesave = strcat('C:/Users/Alpha/Documents/andy/chiral_models_comsol/2step_gammadion_data/edata/structure',num2str(counter),'_ewfd2NormE-',num2str(i),'.txt');
                model.result.export(data).set('filename', filesave);
                model.result.export(data).set('header', false);
                model.result.export(data).run;
            end
            fileStr = strcat('T:/Users/Alpha/Documents/Andy/chiral_data_1of2/',num2str(n1),'/structure',num2str(n2),'.csv');
            structure = importdata(fileStr);
            structure(structure~=2)=0;
            writetable(table(structure), strcat('C:/Users/Alpha/Documents/andy/chiral_models_comsol/2step_gammadion_data/edata/structure',num2str(counter),'.txt'),'WriteVariableNames',false)
            counter = counter + 1;
        catch
            fprintf('\n Error in %d,%d\nData not saved for structure!!!\n',n1,n2)
        end
        
        
    end
end
