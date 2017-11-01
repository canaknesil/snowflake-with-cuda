#include <iostream>
#include "MDUtils.h"

using namespace std;


#define DIM 3



void testKernel()
{

}


void testDevice()
{

}


void testHost()
{
    int i[DIM] = {0};
    int start[] = {2, 2, 2};
    int end[] = {5, 5, 5};

    MDFor(DIM, i, start, end, [&] ()
    {
        for (int a=0; a<DIM; a++) cout << i[a] << " ";
        cout << endl;
    });
}


int main()
{
    testHost();
    testDevice();
    
    return 0;
}