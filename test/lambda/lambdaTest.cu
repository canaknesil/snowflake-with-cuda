#include <iostream>

using namespace std;



__global__
void lambdaTestKernel(int *num)
{
    auto func = [&] () { *num = 5; };
    func();
}


void testDevice()
{
    int num = 0;

    int *d_num;
    cudaMalloc(&d_num, sizeof(int));

    cudaMemcpy(d_num, &num, sizeof(int), cudaMemcpyHostToDevice);

    lambdaTestKernel <<<1, 1>>> (d_num);

    cudaMemcpy(&num, d_num, sizeof(int), cudaMemcpyDeviceToHost);

    cout << num << endl;
}


void testHost()
{
    char str[] = "Another Hello World!";

    void (*func1)() = [] () { cout << "Hello world" << endl; };
    func1();

    auto func2 = [] () { cout << "Hello world" << endl; };
    func2();

    auto func3 = [&] () { cout << str << endl; };
    func3();
}


int main()
{
    testHost();
    testDevice();

    return 0;
}