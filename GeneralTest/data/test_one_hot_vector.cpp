#include "test_one_hot_vector.h"
#include <iostream>
#include <cassert>
#include <MetaNN/meta_nn.h>
using namespace std;
using namespace MetaNN;

namespace
{
void test_one_hot_vector1()
{
    cout << "Test one-hot vector case 1...\t";
    static_assert(IsMatrix<OneHotColVector<int, DeviceTags::CPU>>, "Test Error");
    static_assert(IsMatrix<OneHotColVector<int, DeviceTags::CPU> &>, "Test Error");
    static_assert(IsMatrix<OneHotColVector<int, DeviceTags::CPU> &&>, "Test Error");
    static_assert(IsMatrix<const OneHotColVector<int, DeviceTags::CPU> &>, "Test Error");
    static_assert(IsMatrix<const OneHotColVector<int, DeviceTags::CPU> &&>, "Test Error");

    auto rm = OneHotColVector<int, DeviceTags::CPU>(100, 37);
    assert(rm.RowNum() == 100);
    assert(rm.ColNum() == 1);
    assert(rm.HotPos() == 37);

    auto rm1 = Evaluate(rm);
    for (size_t i=0; i<100; ++i)
    {
        for (size_t j=0; j<1; ++j)
        {
            if (i != 37)
            {
                assert(rm1(i, j) == 0);
            }
            else
            {
                assert(rm1(i, j) == 1);
            }
        }
    }

    cout << "done" << endl;
}

void test_one_hot_vector2()
{
    cout << "Test one-hot vector case 2...\t";
    static_assert(IsMatrix<OneHotRowVector<int, DeviceTags::CPU>>, "Test Error");
    static_assert(IsMatrix<OneHotRowVector<int, DeviceTags::CPU> &>, "Test Error");
    static_assert(IsMatrix<OneHotRowVector<int, DeviceTags::CPU> &&>, "Test Error");
    static_assert(IsMatrix<const OneHotRowVector<int, DeviceTags::CPU> &>, "Test Error");
    static_assert(IsMatrix<const OneHotRowVector<int, DeviceTags::CPU> &&>, "Test Error");

    auto rm = OneHotRowVector<int, DeviceTags::CPU>(100, 37);
    assert(rm.RowNum() == 1);
    assert(rm.ColNum() == 100);
    assert(rm.HotPos() == 37);

    auto rm1 = Evaluate(rm);
    for (size_t i=0; i<1; ++i)
    {
        for (size_t j=0; j<100; ++j)
        {
            if (j != 37)
            {
                assert(rm1(i, j) == 0);
            }
            else
            {
                assert(rm1(i, j) == 1);
            }
        }
    }

    cout << "done" << endl;
}

void test_one_hot_vector3()
{
    cout << "Test one-hot vector case 3...\t";
    auto rm1 = OneHotRowVector<int, DeviceTags::CPU>(100, 37);
    auto rm2 = OneHotRowVector<int, DeviceTags::CPU>(50, 16);
    auto cm1 = OneHotColVector<int, DeviceTags::CPU>(101, 20);
    auto cm2 = OneHotColVector<int, DeviceTags::CPU>(49, 18);

    auto evalRes1 = rm1.EvalRegister();
    auto evalRes2 = rm2.EvalRegister();
    auto evalRes3 = cm1.EvalRegister();
    auto evalRes4 = cm2.EvalRegister();

    EvalPlan<DeviceTags::CPU>::Eval();
    for (size_t j = 0; j < 100; ++j)
    {
        if (j == 37)
        {
            assert(evalRes1.Data()(0, j) == 1);
        }
        else
        {
            assert(evalRes1.Data()(0, j) == 0);
        }
    }

    for (size_t j = 0; j < 50; ++j)
    {
        if (j == 16)
        {
            assert(evalRes2.Data()(0, j) == 1);
        }
        else
        {
            assert(evalRes2.Data()(0, j) == 0);
        }
    }

    for (size_t j = 0; j < 101; ++j)
    {
        if (j == 20)
        {
            assert(evalRes3.Data()(j, 0) == 1);
        }
        else
        {
            assert(evalRes3.Data()(j, 0) == 0);
        }
    }

    for (size_t j = 0; j < 49; ++j)
    {
        if (j == 18)
        {
            assert(evalRes4.Data()(j, 0) == 1);
        }
        else
        {
            assert(evalRes4.Data()(j, 0) == 0);
        }
    }
    cout << "done" << endl;
}
}

void test_one_hot_vector()
{
    test_one_hot_vector1();
    test_one_hot_vector2();
    test_one_hot_vector3();
}
