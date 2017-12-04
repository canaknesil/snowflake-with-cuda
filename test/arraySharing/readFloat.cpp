#include <iostream>
#include <cstdio>

using namespace std;

int main()
{
	FILE *f = fopen("file", "r");
	float arr[5];
	fread(arr, sizeof(float), 5, f);
	fclose(f);

	int i;
	for (i=0; i<5; i++)
	{
		cout << arr[i] << " ";
	}
	cout << endl;

	return 0;
}