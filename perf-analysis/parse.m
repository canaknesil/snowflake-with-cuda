
[side py1 c1 cu1] = textread("run1.txt", "%d %f %f %f");
[dum1 py2 c2 cu2] = textread("run2.txt", "%d %f %f %f");
[dum1 dum2 c3 cu3] = textread("run3.txt", "%d %f %f %f");
[dum1 dum2 c4 cu4] = textread("run4.txt", "%d %f %f %f");

size = side .* side;
py = (py1 + py2) ./ 2;
c = (c1 + c2 + c3 + c4) ./ 4;
cu = (cu1 + cu2 + c3 + c4) ./ 4;

loglog(size, py, '*-b', "linewidth", 4)
hold
loglog(size, cu, '*-r', "linewidth", 4)
loglog(size, c, '*-g', "linewidth", 4)

title('Snowflake Backends Comparison', "fontsize", 24)
xlabel('Input Data Size (# of elements)', "fontsize", 20)
ylabel('Callable execution time (s)', "fontsize", 20)
l = legend('Python Compiler', 'CUDA Compiler', 'OpenCL Compiler');

set(gca, "fontsize", 18)
set(l, "fontsize", 18)

grid on
