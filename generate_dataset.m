dr = 'C:/Users/Alpha/Documents/andy/chiral_models_comsol/2step_gammadion_data/';
logFile = strcat(dr,'dataGeneration_gammadionChiral-2step_',date,'.log');
fid = fopen(logFile, 'w');
fprintf(fid, 'Data generation log for the creation of gammadion structures for the chiral paper (2-addition steps).\nDirectory: %s\n\n',dr);
fclose(fid);
format shortg

c = clock;
rng(round(c(6)*1921),'twister')
f = waitbar(0,'Starting...');
pause(0.5)
for j = 1:500
    waitbar(j/500,f,strcat('Calculating structure #',num2str(j),'...'));
    output = -1;
    fid = fopen(logFile, 'a+');
    fprintf(fid,strcat(num2str(clock),'\n'));
    fprintf(fid, '%3d Attempting to create / evaluate sructure.\n\n',j);
    fclose(fid);
    try
        output = base(dr,j);
    catch e %e is an MException struct
        fprintf(1,'The identifier was:\n%s',e.identifier);
        fprintf(1,'There was an error! The message was:\n%s',e.message);
        % more error handling...
    end
    waitbar(j/500,f,strcat('Saving structure #',num2str(j),'...'));
    if output == -1 || output == 0
        fid = fopen(logFile, 'a+');
        fprintf(fid,strcat(num2str(clock),'\n'));
        fprintf(fid, '%3d Error creating / evaluating COMSOL model. Output code %d\n\n',j,output);
        fclose(fid);
    else
        fid = fopen(logFile, 'a+');
        fprintf(fid,strcat(num2str(clock),'\n'));
        fprintf(fid, '%3d COMSOL model successfully evaluated. Output code %d\n',j,output);
        fclose(fid);
    end
end
f = waitbar(1,'Completed!');

fid = fopen(logFile, 'a+');
fprintf(fid,strcat(num2str(clock),'\n'));
fprintf(fid, 'LOG CLOSED.');
fclose(fid);