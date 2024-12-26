F = figure;
ax=axes(F,'XLim',[-3.5 3.5],'YLim',[-2.5 2.5]);
a_lines = cell(1,Nnum);
p_plots = cell(1,Nnum);
hold on;
for i = 1:Nnum
    a_lines{i}=animatedline(ax,'Color',array_of_collors{i},'MaximumNumPoints',200);
    p_plots{i}=plot(array_of_tables{i}(1,1),array_of_tables{i}(1,2),'or','Color',array_of_collors{i});
end
a0=animatedline(ax,'Color',[0 0 0]);
%[mov(:,:,1,1),map]=rgb2ind(f.cdata,256,'nodither');
k=1;
for i=1:10:numel(array_of_tables{1}(1:N,1))
    for j = 1:Nnum
        x=array_of_tables{j}(i,1);
        y=array_of_tables{j}(i,2);
        p_plots{j}.XData=x;
        p_plots{j}.YData=y;
        addpoints(a_lines{j},x,y);
    end
    pause(0.1);
    M(k) = getframe(F);
    k=k+1;
    drawnow
end
v=VideoWriter('VIDEO2');
v.FrameRate = 40;
open(v);
for j=1:length(M)
    writeVideo(v,M(j));
end
close(v);