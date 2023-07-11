counter=35;
for n1 = 7:18
    for n2 = 1:7
        try
%             file = strcat('T:/Users/Alpha/Documents/Andy/chiral_data_1of2/',num2str(n1),'/basemodel',num2str(n2),'.mph');
            file = strcat('C:/Users/Alpha/Documents/andy/chiral_models_comsol/2step_gammadion_data/',num2str(n1),'/basemodel',num2str(n2),'.mph');
            model = mphload(file);
        catch
            fprintf('\n Set %d: elements %d; coutner %d\n',n1,n2-1,counter)
            break
        end
        try
            model.result.numerical.create('av1', 'AvVolume');
            model.result.numerical('av1').selection.set([8]);
            model.result.numerical('av1').setIndex('expr', 'abs(C_CPL/C0)', 0);
            model.result.numerical('av1').set('data', 'dset2');
            model.result.numerical('av1').setIndex('looplevelinput', 'manualindices', 1);
            model.result.numerical('av1').setIndex('looplevelinput', 'manual', 1);
            model.result.numerical('av1').setIndex('looplevel', [201], 1);
            model.result.numerical.create('av2', 'AvVolume');
            model.result.numerical('av2').selection.set([8 9]);
            model.result.numerical('av2').set('data', 'dset2');
            model.result.numerical('av2').setIndex('looplevelinput', 'manualindices', 1);
            model.result.numerical('av2').setIndex('looplevelindices', 201, 1);
            model.result.numerical('av2').setIndex('expr', 'abs(C_CPL/C0)', 0);
            model.result.numerical.create('av3', 'AvVolume');
            model.result.numerical('av3').selection.set([8 9 10]);
            model.result.numerical('av3').setIndex('expr', 'abs(C_CPL/C0)', 0);
            model.result.numerical('av3').set('data', 'dset2');
            model.result.numerical('av3').setIndex('looplevelinput', 'manualindices', 1);
            model.result.numerical('av3').setIndex('looplevelindices', 201, 1);
            model.result.table.create('tbl1', 'Table');
            model.result.table('tbl1').comments('Global Evaluation 1 (ewfd.Atotal, ewfd.Rtotal, ewfd.Ttotal)');
            model.result.numerical('gev1').set('table', 'tbl1');
            model.result.numerical('gev1').setResult;
            model.result.table.create('tbl2', 'Table');
            model.result.table('tbl2').comments('Volume Average 1 (abs(C_CPL/C0))');
            model.result.numerical('av1').set('table', 'tbl2');
            model.result.numerical('av1').setResult;
            model.result.table.create('tbl3', 'Table');
            model.result.table('tbl3').comments('Volume Average 2 (abs(C_CPL/C0))');
            model.result.numerical('av2').set('table', 'tbl3');
            model.result.numerical('av2').setResult;
            model.result.table.create('tbl4', 'Table');
            model.result.table('tbl4').comments('Volume Average 3 (abs(C_CPL/C0))');
            model.result.numerical('av3').set('table', 'tbl4');
            model.result.numerical('av3').setResult;
            model.result.export.create('data2', 'Data');
            model.result.export('data2').set('data', 'dset2');
            model.result.export.create('tbl1', 'Table');
            model.result.export('tbl1').set('table', 'tbl2');
            model.result.export('tbl1').set('header', false);
            model.result.export('tbl1').set('fullprec', false);

            filesave = strcat('C:/Users/Alpha/Documents/andy/chiral_models_comsol/2step_gammadion_data/avgdata/structure',num2str(counter),'_CCPL-1.txt');
            model.result.export('tbl1').set('filename', filesave);
            model.result.export.create('tbl2', 'Table');
            model.result.export('tbl2').set('table', 'tbl3');
            filesave = strcat('C:/Users/Alpha/Documents/andy/chiral_models_comsol/2step_gammadion_data/avgdata/structure',num2str(counter),'_CCPL-2.txt');
            model.result.export('tbl2').set('filename', filesave);
            model.result.export('tbl2').set('header', false);
            model.result.export('tbl2').set('fullprec', false);
            model.result.export.create('tbl3', 'Table');
            model.result.export('tbl3').set('table', 'tbl4');
            filesave = strcat('C:/Users/Alpha/Documents/andy/chiral_models_comsol/2step_gammadion_data/avgdata/structure',num2str(counter),'_CCPL-3.txt');
            model.result.export('tbl3').set('filename', filesave);
            model.result.export('tbl3').set('fullprec', false);
            model.result.export('tbl3').set('header', false);
            model.result.export('data1').active(false);
            model.result.export('data2').active(false);
            model.result.export('tbl1').set('alwaysask', false);
            model.result.export('tbl1').run;
            model.result.export('tbl1').set('alwaysask', false);
            model.result.export('tbl2').set('alwaysask', false);
            model.result.export('tbl2').run;
            model.result.export('tbl2').set('alwaysask', false);
            model.result.export('tbl3').set('alwaysask', false);
            model.result.export('tbl3').run;
            model.result.export('tbl3').set('alwaysask', false);
            
%             fileStr = strcat('T:/Users/Alpha/Documents/Andy/chiral_data_1of2/',num2str(n1),'/structure',num2str(n2),'.csv');
            fileStr = strcat('C:/Users/Alpha/Documents/andy/chiral_models_comsol/2step_gammadion_data/',num2str(n1),'/structure',num2str(n2),'.csv');
            structure = importdata(fileStr);
            structure(structure~=2)=0;
            writetable(table(structure), strcat('C:/Users/Alpha/Documents/andy/chiral_models_comsol/2step_gammadion_data/avgdata/structure',num2str(counter),'.txt'),'WriteVariableNames',false)
            counter = counter + 1;
        catch
            fprintf('\n Error in %d,%d\nData not saved for structure!!!\n',n1,n2)
        end
        
        
    end
end
