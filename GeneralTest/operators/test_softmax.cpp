#include "test_softmax.h"

#include "../facilities/data_gen.h"
#include <MetaNN/meta_nn.h>
#include <cassert>
#include <iostream>
using namespace MetaNN;
using namespace std;

namespace
{
void test_softmax1()
{
    cout << "Test softmax case 1 ...\t";
    auto rm1 = GenMatrix<float>(1, 20, 0, 0.001f);
    auto t = RowSoftmax(rm1);
    auto t_r = Evaluate(t);

    float sum = 0;
    for (size_t i = 0; i < 20; ++i)
    {
        sum += exp(rm1(0, i));
    }

    for (size_t i = 0; i < 20; ++i)
    {
        assert(fabs(t_r(0, i) - exp(rm1(0, i)) / sum) < 0.0001);
    }

    rm1 = GenMatrix<float>(111, 113, 2, 0.001f);
    rm1 = rm1.SubMatrix(17, 18, 31, 51);
    t = RowSoftmax(rm1);
    t_r = Evaluate(t);

    sum = 0;
    for (size_t i = 0; i < 20; ++i)
    {
        sum += exp(rm1(0, i));
    }

    for (size_t i = 0; i < 20; ++i)
    {
        assert(fabs(t_r(0, i) - exp(rm1(0, i)) / sum) < 0.0001);
    }
    cout << "done" << endl;
}

void test_softmax2()
{
    cout << "Test softmax case 2 ...\t";
    auto rm1 = GenMatrix<float>(20, 1, 0, 0.001f);
    auto t = ColSoftmax(rm1);
    auto t_r = Evaluate(t);

    float sum = 0;
    for (size_t i = 0; i < 20; ++i)
    {
        sum += exp(rm1(i, 0));
    }

    for (size_t i = 0; i < 20; ++i)
    {
        assert(fabs(t_r(i, 0) - exp(rm1(i, 0)) / sum) < 0.0001);
    }

    rm1 = GenMatrix<float>(111, 113, 2, 0.001f);
    rm1 = rm1.SubMatrix(31, 51, 17, 18);
    t = ColSoftmax(rm1);
    t_r = Evaluate(t);

    sum = 0;
    for (size_t i = 0; i < 20; ++i)
    {
        sum += exp(rm1(i, 0));
    }

    for (size_t i = 0; i < 20; ++i)
    {
        assert(fabs(t_r(i, 0) - exp(rm1(i, 0)) / sum) < 0.0001);
    }
    cout << "done" << endl;
}

void test_softmax3()
{
    cout << "Test softmax case 3 ...\t";
    {
        auto rm1 = GenMatrix<float>(20, 1, 0, 0.001f);
        auto res = ColSoftmax(rm1);
        auto res2 = ColSoftmax(rm1);

        assert(res == res2);

        auto cm1 = Evaluate(res);
        auto cm2 = Evaluate(res);
        assert(cm1 == cm2);
    }
    {
        auto rm1 = GenMatrix<float>(20, 1, 0, 0.001f);
        auto res = ColSoftmax(rm1);
        auto res2 = res;

        assert(res == res2);

        const auto& evalHandle1 = res.EvalRegister();
        const auto& evalHandle2 = res2.EvalRegister();
        EvalPlan<DeviceTags::CPU>::Eval();

        auto cm1 = evalHandle1.Data();
        auto cm2 = evalHandle2.Data();
        assert(cm1 == cm2);
    }
    {
        auto rm1 = GenMatrix<float>(1, 20, 0, 0.001f);
        auto res = RowSoftmax(rm1);
        auto res2 = RowSoftmax(rm1);

        assert(res == res2);

        auto cm1 = Evaluate(res);
        auto cm2 = Evaluate(res);
        assert(cm1 == cm2);
    }
    {
        auto rm1 = GenMatrix<float>(1, 20, 0, 0.001f);
        auto res = RowSoftmax(rm1);
        auto res2 = res;

        assert(res == res2);

        const auto& evalHandle1 = res.EvalRegister();
        const auto& evalHandle2 = res2.EvalRegister();
        EvalPlan<DeviceTags::CPU>::Eval();

        auto cm1 = evalHandle1.Data();
        auto cm2 = evalHandle2.Data();
        assert(cm1 == cm2);
    }
    cout << "done" << endl;
}
}

void test_softmax()
{
    test_softmax1();
    test_softmax2();
    test_softmax3();
}
