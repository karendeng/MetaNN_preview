#include <MetaNN/meta_nn.h>
#include "../../facilities/data_gen.h"
#include <cassert>
#include <iostream>
#include <map>
#include <algorithm>
using namespace MetaNN;
using namespace std;

namespace
{
void test_weight_layer1()
{
    cout << "Test weight layer case 1 ...\t";
    using RootLayer = MakeLayer<WeightLayer>;
    static_assert(!RootLayer::IsFeedbackOutput, "Test Error");
    static_assert(!RootLayer::IsUpdate, "Test Error");

    RootLayer layer("root", 1, 2);

    std::map<std::string, Matrix<float, DeviceTags::CPU>> params;

    Matrix<float, DeviceTags::CPU> w(2, 1);
    w.SetValue(0, 0, -0.27f);
    w.SetValue(1, 0, -0.41f);
    params["root"] = w;

    layer.LoadWeights(params);

    Matrix<float, DeviceTags::CPU> input(1, 1);
    input.SetValue(0, 0, 1);

    auto wi = LayerIO::Create().Set<LayerIO>(input);

    LayerNeutralInvariant(layer);

    auto out = layer.FeedForward(wi);
    auto res = Evaluate(out.Get<LayerIO>());
    assert(fabs(res(0, 0) + 0.27f) < 0.001);
    assert(fabs(res(1, 0) + 0.41f) < 0.001);

    auto out_grad = layer.FeedBackward(LayerIO::Create());
    auto fbOut = out_grad.Get<LayerIO>();
    static_assert(is_same<decltype(fbOut), NullParameter>::value, "Test error");

    params.clear();
    layer.SaveWeights(params);
    assert(params.find("root") != params.end());

    LayerNeutralInvariant(layer);

    cout << "done" << endl;
}

void test_weight_layer2()
{
    cout << "Test weight layer case 2 ...\t";
    using RootLayer = MakeLayer<WeightLayer, PRowVecInput>;
    static_assert(!RootLayer::IsFeedbackOutput, "Test Error");
    static_assert(!RootLayer::IsUpdate, "Test Error");

    RootLayer layer("root", 1, 2);
    std::map<std::string, Matrix<float, DeviceTags::CPU>> params;

    Matrix<float, DeviceTags::CPU> w(1, 2);
    w.SetValue(0, 0, -0.27f);
    w.SetValue(0, 1, -0.41f);
    params["root"] = w;

    layer.LoadWeights(params);

    Matrix<float, DeviceTags::CPU> input(1, 1);
    input.SetValue(0, 0, 1);

    LayerNeutralInvariant(layer);
    auto wi = LayerIO::Create().Set<LayerIO>(input);

    auto out = layer.FeedForward(wi);
    auto res = Evaluate(out.Get<LayerIO>());
    assert(fabs(res(0, 0) + 0.27f) < 0.001);
    assert(fabs(res(0, 1) + 0.41f) < 0.001);

    auto out_grad = layer.FeedBackward(LayerIO::Create());
    auto fbOut = out_grad.Get<LayerIO>();
    static_assert(is_same<decltype(fbOut), NullParameter>::value, "Test error");

    params.clear();
    layer.SaveWeights(params);
    assert(params.find("root") != params.end());

    LayerNeutralInvariant(layer);
    cout << "done" << endl;
}

void test_weight_layer3()
{
    cout << "Test weight layer case 3 ...\t";

    using RootLayer = MakeLayer<WeightLayer, PUpdate>;
    static_assert(!RootLayer::IsFeedbackOutput, "Test Error");
    static_assert(RootLayer::IsUpdate, "Test Error");

    RootLayer layer("root", 1, 2);

    std::map<std::string, Matrix<float, DeviceTags::CPU>> params;

    Matrix<float, DeviceTags::CPU> w(2, 1);
    w.SetValue(0, 0, -0.27f);
    w.SetValue(1, 0, -0.41f);
    params["root"] = w;

    layer.LoadWeights(params);

    Matrix<float, DeviceTags::CPU> input(1, 1);
    input.SetValue(0, 0, 0.1f);

    LayerNeutralInvariant(layer);
    auto wi = LayerIO::Create().Set<LayerIO>(input);
    auto out = layer.FeedForward(wi);
    auto res = Evaluate(out.Get<LayerIO>());
    assert(fabs(res(0, 0) + 0.027f) < 0.001);
    assert(fabs(res(1, 0) + 0.041f) < 0.001);

    Matrix<float, DeviceTags::CPU> g(2, 1);
    g.SetValue(0, 0, -0.0495f);
    g.SetValue(1, 0, -0.0997f);
    auto out_grad = layer.FeedBackward(LayerIO::Create().Set<LayerIO>(g));
    auto fbOut = out_grad.Get<LayerIO>();
    static_assert(is_same<decltype(fbOut), NullParameter>::value, "Test error");

    GradCollector<float, DeviceTags::CPU> grad_collector;
    layer.GradCollect(grad_collector);
    assert(grad_collector.size() == 1);

    auto w1 = (*grad_collector.begin()).weight;
    auto info_g = Evaluate(Collapse((*grad_collector.begin()).grad));

    assert(fabs(w1(0, 0) + 0.27f) < 0.001);
    assert(fabs(w1(1, 0) + 0.41f) < 0.001);

    assert(fabs(info_g(0, 0) + 0.00495f) < 0.001);
    assert(fabs(info_g(1, 0) + 0.00997f) < 0.001);

    params.clear();
    layer.SaveWeights(params);
    assert(params.find("root") != params.end());

    LayerNeutralInvariant(layer);
    cout << "done" << endl;
}

void test_weight_layer4()
{
    cout << "Test weight layer case 4 ...\t";
    using RootLayer = MakeLayer<WeightLayer, PUpdate, PRowVecInput>;
    static_assert(!RootLayer::IsFeedbackOutput, "Test Error");
    static_assert(RootLayer::IsUpdate, "Test Error");

    RootLayer layer("root", 1, 2);

    std::map<std::string, Matrix<float, DeviceTags::CPU>> params;

    Matrix<float, DeviceTags::CPU> w(1, 2);
    w.SetValue(0, 0, -0.27f);
    w.SetValue(0, 1, -0.41f);
    params["root"] = w;

    layer.LoadWeights(params);

    Matrix<float, DeviceTags::CPU> input(1, 1);
    input.SetValue(0, 0, 0.1f);

    auto wi = LayerIO::Create().Set<LayerIO>(input);

    LayerNeutralInvariant(layer);
    auto out = layer.FeedForward(wi);
    auto res = Evaluate(out.Get<LayerIO>());
    assert(fabs(res(0, 0) + 0.027f) < 0.001);
    assert(fabs(res(0, 1) + 0.041f) < 0.001);

    Matrix<float, DeviceTags::CPU> g(1, 2);
    g.SetValue(0, 0, -0.0495f);
    g.SetValue(0, 1, -0.0997f);
    auto out_grad = layer.FeedBackward(LayerIO::Create().Set<LayerIO>(g));
    auto fbOut = out_grad.Get<LayerIO>();
    static_assert(is_same<decltype(fbOut), NullParameter>::value, "Test error");

    GradCollector<float, DeviceTags::CPU> grad_collector;
    layer.GradCollect(grad_collector);
    assert(grad_collector.size() == 1);

    auto w1 = (*grad_collector.begin()).weight;
    auto info_g = Evaluate(Collapse((*grad_collector.begin()).grad));

    assert(fabs(w1(0, 0) + 0.27f) < 0.001);
    assert(fabs(w1(0, 1) + 0.41f) < 0.001);

    assert(fabs(info_g(0, 0) + 0.00495f) < 0.001);
    assert(fabs(info_g(0, 1) + 0.00997f) < 0.001);

    params.clear();
    layer.SaveWeights(params);
    assert(params.find("root") != params.end());
    LayerNeutralInvariant(layer);

    cout << "done" << endl;
}

void test_weight_layer5()
{
    cout << "Test weight layer case 5 ...\t";
    using RootLayer = MakeLayer<WeightLayer, PUpdate, PFeedbackOutput>;
    static_assert(RootLayer::IsFeedbackOutput, "Test Error");
    static_assert(RootLayer::IsUpdate, "Test Error");

    RootLayer layer("root", 2, 2);

    std::map<std::string, Matrix<float, DeviceTags::CPU>> params;

    Matrix<float, DeviceTags::CPU> w(2, 2);
    w.SetValue(0, 0, 1.1f); w.SetValue(0, 1, 0.1f);
    w.SetValue(1, 0, 3.1f); w.SetValue(1, 1, 1.17f);
    params["root"] = w;

    layer.LoadWeights(params);

    Matrix<float, DeviceTags::CPU> input(2, 1);
    input.SetValue(0, 0, 0.999f);
    input.SetValue(1, 0, 0.0067f);

    auto wi = LayerIO::Create().Set<LayerIO>(input);

    LayerNeutralInvariant(layer);
    auto out = layer.FeedForward(wi);
    auto res = Evaluate(out.Get<LayerIO>());
    assert(fabs(res(0, 0) - 1.0996f) < 0.001);
    assert(fabs(res(1, 0) - 3.1047f) < 0.001);

    Matrix<float, DeviceTags::CPU> g(2, 1);
    g.SetValue(0, 0, 0.0469f);
    g.SetValue(1, 0, -0.0394f);
    auto out_grad = layer.FeedBackward(LayerIO::Create().Set<LayerIO>(g));
    auto fbOut = Evaluate(out_grad.Get<LayerIO>());
    assert(fabs(fbOut(0, 0) + 0.07055) < 0.001);
    assert(fabs(fbOut(1, 0) + 0.041408f) < 0.001);

    GradCollector<float, DeviceTags::CPU> grad_collector;
    layer.GradCollect(grad_collector);
    assert(grad_collector.size() == 1);

    auto w1 = (*grad_collector.begin()).weight;
    auto info_g = Evaluate(Collapse((*grad_collector.begin()).grad));

    assert(fabs(w1(0, 0) - 1.1) < 0.001);
    assert(fabs(w1(0, 1) - 0.1) < 0.001);
    assert(fabs(w1(1, 0) - 3.1) < 0.001);
    assert(fabs(w1(1, 1) - 1.17) < 0.001);

    assert(fabs(info_g(0, 0) - 0.0468531) < 0.001);
    assert(fabs(info_g(0, 1) - 0.00031423) < 0.001);
    assert(fabs(info_g(1, 0) + 0.0393606) < 0.001);
    assert(fabs(info_g(1, 1) + 0.00026398) < 0.001);

    params.clear();
    layer.SaveWeights(params);
    assert(params.find("root") != params.end());

    LayerNeutralInvariant(layer);
    cout << "done" << endl;
}

void test_weight_layer6()
{
    cout << "Test weight layer case 6 ...\t";
    using RootLayer = MakeLayer<WeightLayer, PUpdate, PFeedbackOutput>;
    static_assert(RootLayer::IsFeedbackOutput, "Test Error");
    static_assert(RootLayer::IsUpdate, "Test Error");

    RootLayer layer("root", 8, 4);

    std::map<std::string, Matrix<float, DeviceTags::CPU>> params;

    auto w = GenMatrix<float>(4, 8, 0.1f, 0.5f);
    params["root"] = w;

    layer.LoadWeights(params);

    vector<Matrix<float, DeviceTags::CPU>> op_in;
    vector<Matrix<float, DeviceTags::CPU>> op_grad;

    for (int loop_count = 0; loop_count < 10; ++loop_count)
    {
        auto input = GenMatrix<float>(8, 1, loop_count * 0.1f, -0.3f);
        op_in.push_back(input);

        auto out = layer.FeedForward(LayerIO::Create().Set<LayerIO>(input));
        auto check = Dot(w, input);

        auto handle1 = out.Get<LayerIO>().EvalRegister();
        auto handle2 = check.EvalRegister();
        EvalPlan<DeviceTags::CPU>::Eval();

        auto res = handle1.Data();
        auto c = handle2.Data();
        assert(res.RowNum() == 4);
        assert(res.ColNum() == 1);

        for (size_t i = 0; i < 4; ++i)
        {
            assert(fabs(res(i, 0) - c(i, 0)) <= 0.0001f);
        }
    }

    for (int loop_count = 9; loop_count >= 0; --loop_count)
    {
        auto grad = GenMatrix<float>(4, 1, loop_count * 0.2f, -0.1f);
        op_grad.push_back(grad);
        auto out_grad = layer.FeedBackward(LayerIO::Create().Set<LayerIO>(grad));
        auto check = Dot(Transpose(w), grad);

        auto handle1 = out_grad.Get<LayerIO>().EvalRegister();
        auto handle2 = check.EvalRegister();
        EvalPlan<DeviceTags::CPU>::Eval();

        auto fbOut = handle1.Data();
        auto aimFbout = handle2.Data();

        assert(fbOut.RowNum() == 8);
        assert(fbOut.ColNum() == 1);

        for (size_t i = 0; i < 8; ++i)
        {
            assert(fabs(fbOut(i, 0) - aimFbout(i, 0)) < 0.0001);
        }
    }
    reverse(op_grad.begin(), op_grad.end());

    GradCollector<float, DeviceTags::CPU> grad_collector;
    layer.GradCollect(grad_collector);
    assert(grad_collector.size() == 1);

    auto w1 = grad_collector.begin()->weight;
    auto aim = Evaluate(Dot(op_grad[0], Transpose(op_in[0])));
    for (int loop_count = 1; loop_count < 10; ++loop_count)
    {
        aim = Evaluate((aim + Dot(op_grad[loop_count], Transpose(op_in[loop_count]))));
    }

    auto info_g = Evaluate(Collapse(grad_collector.begin()->grad));

    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j < 8; ++j)
        {
            assert(fabs(aim(i, j) - info_g(i, j)) < 0.0001f);
        }
    }

    params.clear();
    layer.SaveWeights(params);
    assert(params.find("root") != params.end());

    LayerNeutralInvariant(layer);
    cout << "done" << endl;
}

void test_weight_layer7()
{
    cout << "Test weight layer case 7 ...\t";
    using RootLayer = MakeLayer<WeightLayer, PRowVecInput, PUpdate, PFeedbackOutput>;
    static_assert(RootLayer::IsFeedbackOutput, "Test Error");
    static_assert(RootLayer::IsUpdate, "Test Error");

    RootLayer layer("root", 8, 4);

    std::map<std::string, Matrix<float, DeviceTags::CPU>> params;

    auto w = GenMatrix<float>(8, 4, 0.1f, 0.5f);
    params["root"] = w;

    layer.LoadWeights(params);

    vector<Matrix<float, DeviceTags::CPU>> op_in;
    vector<Matrix<float, DeviceTags::CPU>> op_grad;

    for (int loop_count = 0; loop_count < 10; ++loop_count)
    {
        auto input = GenMatrix<float>(1, 8, loop_count * 0.1f, -0.3f);
        op_in.push_back(input);

        auto out = layer.FeedForward(LayerIO::Create().Set<LayerIO>(input));
        auto check = Dot(input, w);

        auto handle1 = out.Get<LayerIO>().EvalRegister();
        auto handle2 = check.EvalRegister();
        EvalPlan<DeviceTags::CPU>::Eval();

        auto res = handle1.Data();
        assert(res.RowNum() == 1);
        assert(res.ColNum() == 4);
        auto c = handle2.Data();

        for (size_t i = 0; i < 4; ++i)
        {
            assert(fabs(res(0, i) - c(0, i)) <= 0.0001f);
        }
    }

    for (int loop_count = 9; loop_count >= 0; --loop_count)
    {
        auto grad = GenMatrix<float>(1, 4, loop_count * 0.2f, -0.1f);
        op_grad.push_back(grad);
        auto out_grad = layer.FeedBackward(LayerIO::Create().Set<LayerIO>(grad));
        auto check = Dot(grad, Transpose(w));

        auto handle1 = out_grad.Get<LayerIO>().EvalRegister();
        auto handle2 = check.EvalRegister();
        EvalPlan<DeviceTags::CPU>::Eval();

        auto fbOut = handle1.Data();
        auto aimFbout = handle2.Data();
        assert(fbOut.RowNum() == 1);
        assert(fbOut.ColNum() == 8);

        for (size_t i = 0; i < 8; ++i)
        {
            assert(fabs(fbOut(0, i) - aimFbout(0, i)) < 0.0001);
        }
    }
    reverse(op_grad.begin(), op_grad.end());

    GradCollector<float, DeviceTags::CPU> grad_collector;
    layer.GradCollect(grad_collector);
    assert(grad_collector.size() == 1);

    auto w1 = grad_collector.begin()->weight;
    auto aim = Evaluate(Dot(Transpose(op_in[0]), op_grad[0]));
    for (int loop_count = 1; loop_count < 10; ++loop_count)
    {
        aim = Evaluate(aim + Dot(Transpose(op_in[loop_count]), op_grad[loop_count]));
    }

    auto info_g = Evaluate(Collapse(grad_collector.begin()->grad));

    for (size_t i = 0; i < 8; ++i)
    {
        for (size_t j = 0; j < 4; ++j)
        {
            assert(fabs(aim(i, j) - info_g(i, j)) < 0.0001f);
        }
    }

    params.clear();
    layer.SaveWeights(params);
    assert(params.find("root") != params.end());

    LayerNeutralInvariant(layer);
    cout << "done" << endl;
}
}

void test_weight_layer()
{
    test_weight_layer1();
    test_weight_layer2();
    test_weight_layer3();
    test_weight_layer4();
    test_weight_layer5();
    test_weight_layer6();
    test_weight_layer7();
}
