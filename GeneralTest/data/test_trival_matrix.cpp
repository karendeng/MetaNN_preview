#include "test_trival_matrix.h"
#include "../facilities/calculate_tags.h"
#include <iostream>
#include <cassert>
#include <MetaNN/meta_nn.h>
using namespace std;
using namespace MetaNN;

namespace
{
void TestTrivalMatrix1()
{
    cout << "Test trival matrix case 1...\t";
    static_assert(IsMatrix<TrivalMatrix<int, CheckDevice>>, "Test Error");
    static_assert(IsMatrix<TrivalMatrix<int, CheckDevice> &>, "Test Error");
    static_assert(IsMatrix<TrivalMatrix<int, CheckDevice> &&>, "Test Error");
    static_assert(IsMatrix<const TrivalMatrix<int, CheckDevice> &>, "Test Error");
    static_assert(IsMatrix<const TrivalMatrix<int, CheckDevice> &&>, "Test Error");

    auto rm = TrivalMatrix<int, CheckDevice>(10, 20, 100);
    assert(rm.RowNum() == 10);
    assert(rm.ColNum() == 20);

    const auto& evalHandle = rm.EvalRegister();
    EvalPlan<DeviceTags::CPU>::Eval();

    auto rm1 = evalHandle.Data();
    for (size_t i=0; i<10; ++i)
    {
        for (size_t j=0; j<20; ++j)
        {
            assert(rm1(i, j) == 100);
        }
    }

    cout << "done" << endl;
}

void TestTrivalMatrix2()
{
    cout << "Test trival matrix case 2...\t";
    TrivalMatrix<int, CheckDevice> rm1(100, 10, 14);
    TrivalMatrix<int, CheckDevice> rm2(10, 20, 35);

    const auto& evalRes1 = rm1.EvalRegister();
    const auto& evalRes2 = rm2.EvalRegister();

    EvalPlan<DeviceTags::CPU>::Eval();
    for (size_t j = 0; j < 100; ++j)
    {
        for (size_t k = 0; k < 10; ++k)
        {
            assert(evalRes1.Data()(j, k) == 14);
        }
    }

    for (size_t j = 0; j < 10; ++j)
    {
        for (size_t k = 0; k < 20; ++k)
        {
            assert(evalRes2.Data()(j, k) == 35);
        }
    }

    cout << "done" << endl;
}
}

void test_trival_matrix()
{
    TestTrivalMatrix1();
    TestTrivalMatrix2();
}
