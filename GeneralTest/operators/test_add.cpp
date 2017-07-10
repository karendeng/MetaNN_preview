#include "test_add.h"
#include "../facilities/data_gen.h"
#include <MetaNN/meta_nn.h>
#include <cassert>
#include <iostream>
using namespace MetaNN;
using namespace std;

namespace
{
void test_add1()
{
    cout << "Test add case 1 ...\t";
    auto rm1 = GenMatrix<int>(4, 5, 0, 1);
    auto rm2 = GenMatrix<int>(4, 5, 2, 3);
    auto add = rm1 + rm2;
    auto add_r = Evaluate(add);

    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j<5; ++j)
        {
            assert(add_r(i, j) == rm1(i, j) + rm2(i, j));
        }
    }

    rm1 = GenMatrix<int>(111, 113, 1, 2);
    rm2 = GenMatrix<int>(111, 113, 2, 3);
    rm1 = rm1.SubMatrix(31, 35, 17, 22);
    rm2 = rm2.SubMatrix(41, 45, 27, 32);
    add = rm1 + rm2;
    add_r = Evaluate(add);
    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j < 5; ++j)
        {
            assert(add_r(i, j) == rm1(i, j) + rm2(i, j));
        }
    }
    cout << "Done" << endl;
}

void test_add2()
{
    cout << "Test add case 2 ...\t";
    auto rm1 = GenMatrix<int>(4, 5, 0, 1);
    auto add = rm1 + 2;
    auto add_r = Evaluate(add);
    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j<5; ++j)
        {
            assert(add_r(i, j) == rm1(i, j) + 2);
        }
    }

    rm1 = GenMatrix<int>(111, 113, 2, 3);
    rm1 = rm1.SubMatrix(31, 35, 17, 22);
    add = 3 + rm1;
    add_r = Evaluate(add);
    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j<5; ++j)
        {
            assert(add_r(i, j) == rm1(i, j) + 3);
        }
    }
    cout << "Done" << endl;
}

void test_add3()
{
    cout << "Test add case 3 ...\t";
    auto rm1 = TrivalMatrix<int, DeviceTags::CPU>(2, 10, 3);
    auto rm2 = TrivalMatrix<int, DeviceTags::CPU>(2, 10, 5);
    auto add = rm1 + rm2;
    auto add_r = Evaluate(add);
    for (size_t i = 0; i < 2; ++i)
    {
        for (size_t j = 0; j<10; ++j)
        {
            assert(add_r(i, j) == 8);
        }
    }
    cout << "Done" << endl;
}
}

void test_add()
{
    test_add1();
    test_add2();
    test_add3();
}
