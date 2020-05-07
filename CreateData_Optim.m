function state = CreateData_Optim(id)

pu1 = load(sprintf('pu1_%d.2d', id)); % id is for every run in parallel  

[~,~,p,q,~,~] = gnuplot(pu1);
% pressure = p(:,floor(end/2)); flow = q(:,floor(end/2));
pressure = p(:,1); flow = q(:,floor(end/2));

state = [flow; pressure];

end