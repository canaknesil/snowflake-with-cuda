#include "MDUtils.h"



void MDFor(int dim, int *i, int *start, int *end, std::function<void ()> body)
{
    if (dim == 0) 
    {
        body();
        return;
    }

    for (i[0] = start[0]; i[0] < end[0]; i[0]++)
    {
        MDFor(dim - 1, i + 1, start + 1, end + 1, body);
    }
}
