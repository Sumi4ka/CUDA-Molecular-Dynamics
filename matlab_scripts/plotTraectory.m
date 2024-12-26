[path, ~, ~] = fileparts(pwd);
path = path + "\results\";
Nnum = 2;
Nstr=num2str(Nnum);
path = path + "results2Nanotube1";
path = path + "\" + "GPU_Results";
array_of_tables = cell(1, Nnum);
for i = 1:Nnum
    % Создание таблицы для каждого значения i
    array_of_tables{i} = table2array(readtable(path + "\"+ num2str(i) +"Molecule.txt"));
end
a=1;
M1=0;
if (a==1)
N=size(array_of_tables{1},1);
if (M1~=0) 
    N=M1;
end
hold on;
array_of_collors = cell(1,Nnum);
array_of_collors1 = {[0 0 1], [0 1 0], [0 1 1], [1 0 0], [1 0 1], [1 1 0], [0.5 0.5 0.5], [0 0 0]};
for i=1:Nnum
    if i<=8
        array_of_collors{i} = array_of_collors1{i};
    else
        array_of_collors{i} = rand(1, 3);
    end
end
for i = 1:Nnum
    plot3(array_of_tables{i}(1:N,1),array_of_tables{i}(1:N,2),array_of_tables{i}(1:N,3),'Color',array_of_collors{i});
    scatter3(array_of_tables{i}(1,1),array_of_tables{i}(1,2),array_of_tables{i}(1,3),'Color',array_of_collors{i});
end
xlim([-2.5 2.5]);
ylim([-2.5 2.5]);
zlim([-2.5 2.5]);
end
if (a==2)
    subplot(1,3,1);
    plot(1:size(R1,1),R1(:,1));
    subplot(1,3,2);
    plot(1:size(R1,1),R1(:,2));
    subplot(1,3,3);
    plot(1:size(R1,1),R1(:,3));
end
x0=0.357595;