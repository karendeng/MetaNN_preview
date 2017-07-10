#include <MetaNN/meta_nn.h>
#include "../../facilities/data_gen.h"
#include <cassert>
#include <iostream>
using namespace MetaNN;
using namespace std;

namespace
{
void test_softmax_layer1()
{
    cout << "Test softmax layer case 1 ...\t";
    using RootLayer = MakeLayer<SoftmaxLayer>;
    static_assert(!RootLayer::IsFeedbackOutput, "Test Error");
    static_assert(!RootLayer::IsUpdate, "Test Error");

    RootLayer layer;

    Matrix<float, DeviceTags::CPU> in(2, 1);
    in.SetValue(0, 0, -0.27f);
    in.SetValue(1, 0, -0.41f);

    auto input = LayerIO::Create().Set<LayerIO>(in);

    LayerNeutralInvariant(layer);

    auto out = layer.FeedForward(input);
    auto check = ColSoftmax(in);

    auto handle1 = out.Get<LayerIO>().EvalRegister();
    auto handle2 = check.EvalRegister();
    EvalPlan<DeviceTags::CPU>::Eval();

    auto res = handle1.Data();
    auto c = handle2.Data();
    assert(fabs(res(0, 0) - c(0, 0)) < 0.001);
    assert(fabs(res(1, 0) - c(1, 0)) < 0.001);

    NullParameter fbIn;
    auto out_grad = layer.FeedBackward(fbIn);
    auto fb1 = out_grad.Get<LayerIO>();
    static_assert(std::is_same<decltype(fb1), NullParameter>::value, "Test error");

    LayerNeutralInvariant(layer);

    cout << "done" << endl;
}

void test_softmax_layer2()
{
    cout << "Test softmax layer case 2 ...\t";
    using RootLayer = MakeLayer<SoftmaxLayer, PFeedbackOutput>;
    static_assert(RootLayer::IsFeedbackOutput, "Test Error");
    static_assert(!RootLayer::IsUpdate, "Test Error");

    RootLayer layer;

    Matrix<float, DeviceTags::CPU> in(2, 1);
    in.SetValue(0, 0, -0.27f);
    in.SetValue(1, 0, -0.41f);

    auto input = LayerIO::Create().Set<LayerIO>(in);

    LayerNeutralInvariant(layer);
    auto out = layer.FeedForward(input);
    auto check = ColSoftmax(in);

    auto handle1 = out.Get<LayerIO>().EvalRegister();
    auto handle2 = check.EvalRegister();
    EvalPlan<DeviceTags::CPU>::Eval();

    auto res = handle1.Data();
    auto c = handle2.Data();

    assert(fabs(res(0, 0) - c(0, 0)) < 0.001);
    assert(fabs(res(1, 0) - c(1, 0)) < 0.001);

    Matrix<float, DeviceTags::CPU> grad(2, 1);
    grad.SetValue(0, 0, 0.1f);
    grad.SetValue(1, 0, 0.3f);

    auto out_grad = layer.FeedBackward(LayerIO::Create().Set<LayerIO>(grad));
    auto fb = Evaluate(out_grad.Get<LayerIO>());

    c = Evaluate(ColSoftmaxDerivative(grad, c));
    assert(fabs(fb(0, 0) - c(0, 0)) < 0.001);
    assert(fabs(fb(1, 0) - c(1, 0)) < 0.001);

    LayerNeutralInvariant(layer);
    cout << "done" << endl;
}

void test_softmax_layer3()
{
    cout << "Test softmax layer case 3 ...\t";
    using RootLayer = MakeLayer<SoftmaxLayer, PFeedbackOutput, PRowVecInput>;
    static_assert(RootLayer::IsFeedbackOutput, "Test Error");
    static_assert(!RootLayer::IsUpdate, "Test Error");

    RootLayer layer;

    Matrix<float, DeviceTags::CPU> in(1, 2);
    in.SetValue(0, 0, -0.27f);
    in.SetValue(0, 1, -0.41f);

    auto input = LayerIO::Create().Set<LayerIO>(in);

    LayerNeutralInvariant(layer);

    auto out = layer.FeedForward(input);
    auto check = RowSoftmax(in);

    auto handle1 = out.Get<LayerIO>().EvalRegister();
    auto handle2 = check.EvalRegister();
    EvalPlan<DeviceTags::CPU>::Eval();

    auto res = handle1.Data();
    auto c = handle2.Data();

    assert(fabs(res(0, 0) - c(0, 0)) < 0.001);
    assert(fabs(res(0, 1) - c(0, 1)) < 0.001);

    Matrix<float, DeviceTags::CPU> grad(1, 2);
    grad.SetValue(0, 0, 0.1f);
    grad.SetValue(0, 1, 0.3f);

    auto out_grad = layer.FeedBackward(LayerIO::Create().Set<LayerIO>(grad));
    auto fb = Evaluate(out_grad.Get<LayerIO>());

    c = Evaluate(RowSoftmaxDerivative(grad, c));
    assert(fabs(fb(0, 0) - c(0, 0)) < 0.001);
    assert(fabs(fb(0, 1) - c(0, 1)) < 0.001);

    LayerNeutralInvariant(layer);

    cout << "done" << endl;
}

void test_softmax_layer4()
{
    cout << "Test softmax layer case 4 ...\t";
    using RootLayer = MakeLayer<SoftmaxLayer, PFeedbackOutput>;
    static_assert(RootLayer::IsFeedbackOutput, "Test Error");
    static_assert(!RootLayer::IsUpdate, "Test Error");

    RootLayer layer;

    vector<Matrix<float, DeviceTags::CPU>> op;

    LayerNeutralInvariant(layer);
    for (size_t loop_count = 1; loop_count < 10; ++loop_count)
    {
        auto in = GenMatrix<float>(loop_count, 1, 0.1f, 0.13f);

        auto input = LayerIO::Create().Set<LayerIO>(in);

        auto out = layer.FeedForward(input);
        auto check = ColSoftmax(in);

        auto handle1 = out.Get<LayerIO>().EvalRegister();
        auto handle2 = check.EvalRegister();
        EvalPlan<DeviceTags::CPU>::Eval();

        auto res = handle1.Data();
        auto c = handle2.Data();
        op.push_back(c);
        for (size_t i = 0; i < loop_count; ++i)
        {
            assert(fabs(res(i, 0) - c(i, 0)) < 0.0001);
        }
    }

    for (size_t loop_count = 9; loop_count >= 1; --loop_count)
    {
        auto grad = GenMatrix<float>(loop_count, 1, 1, 1.1f);
        auto out_grad = layer.FeedBackward(LayerIO::Create().Set<LayerIO>(grad));
        auto check = ColSoftmaxDerivative(grad, op.back());

        auto handle1 = out_grad.Get<LayerIO>().EvalRegister();
        auto handle2 = check.EvalRegister();
        EvalPlan<DeviceTags::CPU>::Eval();

        auto fb = handle1.Data();
        auto c = handle2.Data();

        op.pop_back();

        for (size_t i = 0; i < loop_count; ++i)
        {
            assert(fabs(fb(i, 0) - c(i, 0)) < 0.0001);
        }
    }

    LayerNeutralInvariant(layer);

    cout << "done" << endl;
}

void test_softmax_layer5()
{
    cout << "Test softmax layer case 5 ...\t";
    using RootLayer = MakeLayer<SoftmaxLayer, PFeedbackOutput, PRowVecInput>;
    static_assert(RootLayer::IsFeedbackOutput, "Test Error");
    static_assert(!RootLayer::IsUpdate, "Test Error");

    RootLayer layer;

    vector<Matrix<float, DeviceTags::CPU>> op;

    LayerNeutralInvariant(layer);
    for (size_t loop_count = 1; loop_count < 10; ++loop_count)
    {
        auto in = GenMatrix<float>(1, loop_count, 0.1f, 0.13f);

        auto input = LayerIO::Create().Set<LayerIO>(in);

        auto out = layer.FeedForward(input);
        auto check = RowSoftmax(in);

        auto handle1 = out.Get<LayerIO>().EvalRegister();
        auto handle2 = check.EvalRegister();
        EvalPlan<DeviceTags::CPU>::Eval();

        auto res = handle1.Data();
        auto c = handle2.Data();

        op.push_back(c);
        for (size_t i = 0; i < loop_count; ++i)
        {
            assert(fabs(res(0, i) - c(0, i)) < 0.0001);
        }
    }

    for (size_t loop_count = 9; loop_count >= 1; --loop_count)
    {
        auto grad = GenMatrix<float>(1, loop_count, 1.3f, 1.1f);
        auto out_grad = layer.FeedBackward(LayerIO::Create().Set<LayerIO>(grad));
        auto check = RowSoftmaxDerivative(grad, op.back());

        auto handle1 = out_grad.Get<LayerIO>().EvalRegister();
        auto handle2 = check.EvalRegister();
        EvalPlan<DeviceTags::CPU>::Eval();

        auto fb = handle1.Data();
        auto c = handle2.Data();
        op.pop_back();

        for (size_t i = 0; i < loop_count; ++i)
        {
            assert(fabs(fb(0, i) - c(0, i)) < 0.0001);
        }
    }

    LayerNeutralInvariant(layer);

    cout << "done" << endl;
}
}

void test_softmax_layer()
{
    test_softmax_layer1();
    test_softmax_layer2();
    test_softmax_layer3();
    test_softmax_layer4();
    test_softmax_layer5();
}
