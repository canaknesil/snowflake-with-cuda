#include "MDUtils.h"



void MDForHost(int dim, int *i, int *start, int *end, std::function<void ()> body)
{
    if (dim == 0) 
    {
        body();
        return;
    }

    for (i[0] = start[0]; i[0] < end[0]; i[0]++)
    {
        MDForHost(dim - 1, i + 1, start + 1, end + 1, body);
    }
}

/*
void MDForDevice(int dim, int *i, int *start, int *end, void (*body)(void **args), void **args)
{
    if (dim == 0) 
    {
        body(args);
        return;
    }

    for (i[0] = start[0]; i[0] < end[0]; i[0]++)
    {
        MDForDevice(dim - 1, i + 1, start + 1, end + 1, body, args);
    }
}
*/