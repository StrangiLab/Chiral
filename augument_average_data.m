clearvars;
direct = 'C:/Users/Alpha/Documents/andy/chiral_models_comsol/2step_gammadion_data/avgdata/';
printCounter = 0;
counter =0;
cd(direct);
mkdir(strcat(direct,'../newavgdata/'));
files = dir('**');
files(1:2) = [];
totalFiles = numel(files);
j=1;
allOutput = zeros(100*4,3);
allInput = zeros(100*4,21,21);
inpCounter=1;
outCounter=1;
loop=1;

real = zeros(21,21);
real(4:6,4:12) = 1;
real(10:18,4:6) = 1;
real(16:18,10:18) = 1;
real(4:12,16:18) = 1;
real(10:12,7:15) = 1;
real(7:15,10:12) = 1;

while j<=totalFiles
    fname = files(j).name;
    filename=strcat(direct,fname);
    if length(fname)>18
        mat = import1dfile(filename);    
        allOutput(outCounter:outCounter+3,loop) = [mat mat mat mat];
        loop = loop+1;
    else
        mat = importdata(filename);
        for l=1:4
            mat = rot90(mat);
            allInput(inpCounter,:,:) = mat-2*real;
            
            inpCounter = inpCounter +1;
        end
    end
    
    if mod(j,4)==0
        outCounter = outCounter + 4;
        loop=1;
    end
    j = j+1;
end

writetable(table(allOutput), strcat(direct,'../newavgdata/allOutput_avgCPL-above.txt'),'WriteVariableNames',false);
% writetable(table(allInput), strcat(direct,'../newavgdata/allInputStructures.txt'),'WriteVariableNames',false);

save(strcat(direct,'../newavgdata/allInputStructuresSubtracted.mat'),'allInput')




function structure0CCPL1 = import1dfile(filename, startRow, endRow)
%IMPORTFILE Import numeric data from a text file as a matrix.
%   STRUCTURE0CCPL1 = IMPORTFILE(FILENAME) Reads data from text file
%   FILENAME for the default selection.
%
%   STRUCTURE0CCPL1 = IMPORTFILE(FILENAME, STARTROW, ENDROW) Reads data
%   from rows STARTROW through ENDROW of text file FILENAME.
%
% Example:
%   structure0CCPL1 = importfile('structure0_CCPL-1.txt', 1, 1);
%
%    See also TEXTSCAN.

% Auto-generated by MATLAB on 2021/08/02 13:17:36

%% Initialize variables.
if nargin<=2
    startRow = 1;
    endRow = 1;
end

%% Format for each line of text:
%   column3: double (%f)
% For more information, see the TEXTSCAN documentation.
formatSpec = '%*58s%f%[^\n\r]';

%% Open the text file.
fileID = fopen(filename,'r');

%% Read columns of data according to the format.
% This call is based on the structure of the file used to generate this
% code. If an error occurs for a different file, try regenerating the code
% from the Import Tool.
dataArray = textscan(fileID, formatSpec, endRow(1)-startRow(1)+1, 'Delimiter', '', 'WhiteSpace', '', 'TextType', 'string', 'HeaderLines', startRow(1)-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');
for block=2:length(startRow)
    frewind(fileID);
    dataArrayBlock = textscan(fileID, formatSpec, endRow(block)-startRow(block)+1, 'Delimiter', '', 'WhiteSpace', '', 'TextType', 'string', 'HeaderLines', startRow(block)-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');
    dataArray{1} = [dataArray{1};dataArrayBlock{1}];
end

%% Close the text file.
fclose(fileID);

%% Post processing for unimportable data.
% No unimportable data rules were applied during the import, so no post
% processing code is included. To generate code which works for
% unimportable data, select unimportable cells in a file and regenerate the
% script.

%% Create output variable
structure0CCPL1 = [dataArray{1:end-1}];

end
