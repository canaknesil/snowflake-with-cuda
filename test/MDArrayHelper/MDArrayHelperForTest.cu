#include <iostream>
#include "MDArrayHelper.h"

using namespace std;


int main()
{
	int dim = 2;
	int dimSize[] = {3, 3};
	int linSize = 9;

	int orjArr[9];

	MDArrayHelper<int> arr(orjArr, dim, dimSize);

	for (int i=0; i<linSize; i++)
	{
		int index[2];
		arr.getCoords(index, i);

		arr.set(index[0] + index[1], index);

		cout << arr.get(index) << " ";
	}

	cout << endl;

	return 0;
}