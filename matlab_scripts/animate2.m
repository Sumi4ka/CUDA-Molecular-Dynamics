F = figure;
ax=axes(F,'XLim',[-2.5 2.5],'YLim',[-2.5 2.5]);
hold on
for i = 1:Nnum
    plot(array_of_tables{i}(1:N,1),array_of_tables{i}(1:N,2),'Color',array_of_collors{i});
    scatter(array_of_tables{i}(1,1),array_of_tables{i}(1,2),'Color',array_of_collors{i});
end
xlim([-2.5 2.5]);
ylim([-2.5 2.5]);
ph_array = cell(1,Nnum);
for i = 1:Nnum
    ph_array{i} = plot(array_of_tables{i}(1,1),array_of_tables{i}(1,2),'ro','MarkerSize',5,'MarkerFaceColor',array_of_collors{i});
end
%[mov(:,:,1,1),map]=rgb2ind(f.cdata,256,'nodither');
k=1;
for i=1:N
    for i1 = 1:Nnum
        ph_array{i1}.XData=array_of_tables{i1}(i,1);
        ph_array{i1}.YData=array_of_tables{i1}(i,2);
    end
    pause(0.000001);
    M(k) = getframe(F);
    k=k+1;
end
v=VideoWriter('VIDEO1');
v.FrameRate = 40;
open(v);
for j=1:length(M)
    writeVideo(v,M(j));
end
close(v);