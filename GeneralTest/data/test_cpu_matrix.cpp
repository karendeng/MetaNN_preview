#include "test_cpu_matrix.h"
#include <iostream>
#include <cassert>
#include <set>
#include <MetaNN/meta_nn.h>
using namespace std;
using namespace MetaNN;

namespace
{
void TestMatrix1()
{
    cout << "Test cpu matrix case 1...\t";
    static_assert(IsMatrix<Matrix<int, DeviceTags::CPU>>, "Test Error");
    static_assert(IsMatrix<Matrix<int, DeviceTags::CPU> &>, "Test Error");
    static_assert(IsMatrix<Matrix<int, DeviceTags::CPU> &&>, "Test Error");
    static_assert(IsMatrix<const Matrix<int, DeviceTags::CPU> &>, "Test Error");
    static_assert(IsMatrix<const Matrix<int, DeviceTags::CPU> &&>, "Test Error");

    Matrix<int, DeviceTags::CPU> rm;
    assert(rm.RowNum() == 0);
    assert(rm.ColNum() == 0);

    rm = Matrix<int, DeviceTags::CPU>(10, 20);
    assert(rm.RowNum() == 10);
    assert(rm.ColNum() == 20);

    int c = 0;
    for (size_t i=0; i<10; ++i)
    {
        for (size_t j=0; j<20; ++j)
        {
            rm.SetValue(i, j, c++);
        }
    }

    const Matrix<int, DeviceTags::CPU> rm2 = rm;
    c = 0;
    for (size_t i=0; i<10; ++i)
    {
        for (size_t j=0; j<20; ++j)
            assert(rm2(i, j) == c++);
    }

    auto rm3 = rm.SubMatrix(3, 7, 5, 15);
    for (size_t i=0; i<rm3.RowNum(); ++i)
    {
        for (size_t j = 0; j<rm3.ColNum(); ++j)
        {
            assert(rm3(i, j) == rm(i+3, j+5));
        }
    }

    auto evalHandle = rm.EvalRegister();
    auto cm = evalHandle.Data();

    for (size_t i=0; i<cm.RowNum(); ++i)
    {
        for (size_t j = 0; j<cm.ColNum(); ++j)
        {
            assert(cm(i, j) == rm(i, j));
        }
    }
    cout << "done" << endl;
}

void TestMatrix2()
{
    cout << "Test cpu matrix case 2...\t";
    auto rm1 = Matrix<int, DeviceTags::CPU>(10, 20);
    int c = 0;
    for (size_t i = 0; i < 10; ++i)
    {
        for (size_t j = 0; j < 20; ++j)
        {
            rm1.SetValue(i, j, c++);
        }
    }

    auto rm2 = Matrix<int, DeviceTags::CPU>(3, 7);
    for (size_t i = 0; i < 3; ++i)
    {
        for (size_t j = 0; j < 7; ++j)
        {
            rm2.SetValue(i, j, c++);
        }
    }
    cout << "done" << endl;
}

void TestMatrix3()
{
    cout << "Test cpu matrix case 3...\t";
    static_assert(IsBatchMatrix<Batch<Matrix<int, DeviceTags::CPU>>>, "Test Error");
    static_assert(IsBatchMatrix<Batch<Matrix<int, DeviceTags::CPU>> &>, "Test Error");
    static_assert(IsBatchMatrix<Batch<Matrix<int, DeviceTags::CPU>> &&>, "Test Error");
    static_assert(IsBatchMatrix<const Batch<Matrix<int, DeviceTags::CPU>> &>, "Test Error");
    static_assert(IsBatchMatrix<const Batch<Matrix<int, DeviceTags::CPU>> &&>, "Test Error");

    auto rm1 = Batch<Matrix<int, DeviceTags::CPU>>(10, 20);
    assert(rm1.BatchNum() == 0);
    assert(rm1.IsEmpty());

    int c = 0;
    auto me1 = Matrix<int, DeviceTags::CPU>(10, 20);
    auto me2 = Matrix<int, DeviceTags::CPU>(10, 20);
    auto me3 = Matrix<int, DeviceTags::CPU>(10, 20);
    for (size_t i = 0; i < 10; ++i)
    {
        for (size_t j = 0; j < 20; ++j)
        {
            me1.SetValue(i, j, c++);
            me2.SetValue(i, j, c++);
            me3.SetValue(i, j, c++);
        }
    }
    rm1.PushBack(me1);
    rm1.PushBack(me2);
    rm1.PushBack(me3);
    assert(rm1.BatchNum() == 3);
    assert(!rm1.IsEmpty());

    for (size_t i = 0; i < 10; ++i)
    {
        for (size_t j = 0; j < 20; ++j)
        {
            assert(rm1[0](i, j) == me1(i, j));
            assert(rm1[1](i, j) == me2(i, j));
            assert(rm1[2](i, j) == me3(i, j));
        }
    }

    rm1 = rm1.SubMatrix(3, 7, 11, 16);
    assert(rm1.RowNum() == 4);
    assert(rm1.ColNum() == 5);
    assert(rm1.BatchNum() == 3);
    me1 = me1.SubMatrix(3, 7, 11, 16);
    me2 = me2.SubMatrix(3, 7, 11, 16);
    me3 = me3.SubMatrix(3, 7, 11, 16);
    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j < 5; ++j)
        {
            assert(rm1[0](i, j) == me1(i, j));
            assert(rm1[1](i, j) == me2(i, j));
            assert(rm1[2](i, j) == me3(i, j));
        }
    }

    auto evalHandle = rm1.EvalRegister();
    EvalPlan<DeviceTags::CPU>::Eval();
    auto rm2 = evalHandle.Data();

    for (size_t k = 0; k < 3; ++k)
    {
        for (size_t i = 0; i < 4; ++i)
        {
            for (size_t j = 0; j < 5; ++j)
            {
                assert(rm1[k](i, j) == rm2[k](i, j));
            }
        }
    }
    cout << "done" << endl;
}
}

void test_cpu_matrix()
{
    TestMatrix1();
    TestMatrix2();
    TestMatrix3();
}