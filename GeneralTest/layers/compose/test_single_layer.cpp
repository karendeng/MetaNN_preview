#include <MetaNN/meta_nn.h>
#include "../../facilities/data_gen.h"
#include <cassert>
#include <map>
#include <iostream>
using namespace MetaNN;
using namespace std;

namespace
{
void test_single_layer1()
{
    // No update, action is sigmoid, with bias
    cout << "Test single layer case 1 ...\t";
    using RootLayer = MakeLayer<SingleLayer>;
    static_assert(!RootLayer::IsUpdate, "Test Error");
    static_assert(!RootLayer::IsFeedbackOutput, "Test Error");

    RootLayer layer("root", 2, 3);
    map<string, Matrix<float, DeviceTags::CPU>> params;

    Matrix<float, DeviceTags::CPU> w1(3, 2);
    w1.SetValue(0, 0, 0.1f);  w1.SetValue(0, 1, 0.2f);
    w1.SetValue(1, 0, 0.3f);  w1.SetValue(1, 1, 0.4f);
    w1.SetValue(2, 0, 0.5f);  w1.SetValue(2, 1, 0.6f);
    params["root-weight"] = w1;

    Matrix<float, DeviceTags::CPU> b1(3, 1);
    b1.SetValue(0, 0, 0.7f);  b1.SetValue(1, 0, 0.8f); b1.SetValue(2, 0, 0.9f);
    params["root-bias"] = b1;

    LayerLoadWeights(layer, params);
//    layer.LoadWeights(params);

    Matrix<float, DeviceTags::CPU> i(2, 1);
    i.SetValue(0, 0, 0.1f); i.SetValue(1, 0, 0.2f);

    auto input = LayerIO::Create().Set<LayerIO>(i);
    auto out = Evaluate(layer.FeedForward(input).Get<LayerIO>());

    assert(fabs(out(0, 0) - (1 / (1+exp(-0.75)))) < 0.00001);
    assert(fabs(out(1, 0) - (1 / (1+exp(-0.91)))) < 0.00001);
    assert(fabs(out(2, 0) - (1 / (1+exp(-1.07)))) < 0.00001);

    auto fbIn = LayerIO::Create();
    auto out_grad = layer.FeedBackward(fbIn);
    auto fbOut = out_grad.Get<LayerIO>();
    static_assert(is_same<decltype(fbOut), NullParameter>::value, "Test error");

    params.clear();
    layer.SaveWeights(params);
    assert(params.find("root-weight") != params.end());
    assert(params.find("root-bias") != params.end());

    cout << "done" << endl;
}

void test_single_layer2()
{
    // No update, action is tanh, with bias
    cout << "Test single layer case 2 ...\t";
    using RootLayer = MakeLayer<SingleLayer, PTanhAction>;
    static_assert(!RootLayer::IsUpdate, "Test Error");
    static_assert(!RootLayer::IsFeedbackOutput, "Test Error");

    RootLayer layer("root", 2, 3);
    map<string, Matrix<float, DeviceTags::CPU>> params;

    Matrix<float, DeviceTags::CPU> w1(3, 2);
    w1.SetValue(0, 0, 0.1f);  w1.SetValue(0, 1, 0.2f);
    w1.SetValue(1, 0, 0.3f);  w1.SetValue(1, 1, 0.4f);
    w1.SetValue(2, 0, 0.5f);  w1.SetValue(2, 1, 0.6f);
    params["root-weight"] = w1;

    Matrix<float, DeviceTags::CPU> b1(3, 1);
    b1.SetValue(0, 0, 0.7f);  b1.SetValue(1, 0, 0.8f); b1.SetValue(2, 0, 0.9f);
    params["root-bias"] = b1;

    layer.LoadWeights(params);

    Matrix<float, DeviceTags::CPU> i(2, 1);
    i.SetValue(0, 0, 0.1f); i.SetValue(1, 0, 0.2f);

    auto input = LayerIO::Create().Set<LayerIO>(i);
    auto out = Evaluate(layer.FeedForward(input).Get<LayerIO>());

    assert(fabs(out(0, 0) - tanh(0.75)) < 0.00001);
    assert(fabs(out(1, 0) - tanh(0.91)) < 0.00001);
    assert(fabs(out(2, 0) - tanh(1.07)) < 0.00001);

    auto out_grad = layer.FeedBackward(LayerIO::Create());
    auto fbOut = out_grad.Get<LayerIO>();
    static_assert(is_same<decltype(fbOut), NullParameter>::value, "Test error");

    params.clear();
    layer.SaveWeights(params);
    assert(params.find("root-weight") != params.end());
    assert(params.find("root-bias") != params.end());

    cout << "done" << endl;
}

void test_single_layer3()
{
    // No update, action is sigmoid, no bias
    cout << "Test single layer case 3 ...\t";
    using RootLayer = MakeLayer<SingleLayer, PNoBiasSingleLayer>;
    static_assert(!RootLayer::IsUpdate, "Test Error");
    static_assert(!RootLayer::IsFeedbackOutput, "Test Error");

    RootLayer layer("root", 2, 3);
    map<string, Matrix<float, DeviceTags::CPU>> params;

    Matrix<float, DeviceTags::CPU> w1(3, 2);
    w1.SetValue(0, 0, 0.1f);  w1.SetValue(0, 1, 0.2f);
    w1.SetValue(1, 0, 0.3f);  w1.SetValue(1, 1, 0.4f);
    w1.SetValue(2, 0, 0.5f);  w1.SetValue(2, 1, 0.6f);
    params["root-weight"] = w1;

    layer.LoadWeights(params);

    Matrix<float, DeviceTags::CPU> i(2, 1);
    i.SetValue(0, 0, 0.1f); i.SetValue(1, 0, 0.2f);

    auto input = LayerIO::Create().Set<LayerIO>(i);
    auto out = Evaluate(layer.FeedForward(input).Get<LayerIO>());

    assert(fabs(out(0, 0) - (1 / (1+exp(-0.05)))) < 0.00001);
    assert(fabs(out(1, 0) - (1 / (1+exp(-0.11)))) < 0.00001);
    assert(fabs(out(2, 0) - (1 / (1+exp(-0.17)))) < 0.00001);

    auto out_grad = layer.FeedBackward(LayerIO::Create());
    auto fbOut = out_grad.Get<LayerIO>();
    static_assert(is_same<decltype(fbOut), NullParameter>::value, "Test error");

    params.clear();
    layer.SaveWeights(params);
    assert(params.find("root-weight") != params.end());

    cout << "done" << endl;
}

void test_single_layer4()
{
    // Update, action is sigmoid, with bias
    cout << "Test single layer case 4 ...\t";
    using RootLayer = MakeLayer<SingleLayer, PUpdate>;
    static_assert(RootLayer::IsUpdate, "Test Error");
    static_assert(!RootLayer::IsFeedbackOutput, "Test Error");

    RootLayer layer("root", 2, 3);
    map<string, Matrix<float, DeviceTags::CPU>> params;

    Matrix<float, DeviceTags::CPU> w1(3, 2);
    w1.SetValue(0, 0, 0.1f);  w1.SetValue(0, 1, 0.2f);
    w1.SetValue(1, 0, 0.3f);  w1.SetValue(1, 1, 0.4f);
    w1.SetValue(2, 0, 0.5f);  w1.SetValue(2, 1, 0.6f);
    params["root-weight"] = w1;

    Matrix<float, DeviceTags::CPU> b1(3, 1);
    b1.SetValue(0, 0, 0.7f);  b1.SetValue(1, 0, 0.8f); b1.SetValue(2, 0, 0.9f);
    params["root-bias"] = b1;

    layer.LoadWeights(params);

    Matrix<float, DeviceTags::CPU> i(2, 1);
    i.SetValue(0, 0, 0.1f); i.SetValue(1, 0, 0.2f);

    auto input = LayerIO::Create().Set<LayerIO>(i);
    auto out = Evaluate(layer.FeedForward(input).Get<LayerIO>());

    assert(fabs(out(0, 0) - (1 / (1+exp(-0.75)))) < 0.00001);
    assert(fabs(out(1, 0) - (1 / (1+exp(-0.91)))) < 0.00001);
    assert(fabs(out(2, 0) - (1 / (1+exp(-1.07)))) < 0.00001);

    Matrix<float, DeviceTags::CPU> grad(3, 1);
    grad.SetValue(0, 0, 0.19);
    grad.SetValue(1, 0, 0.23);
    grad.SetValue(2, 0, -0.15);

    auto fbIn = LayerIO::Create().Set<LayerIO>(grad);
    auto out_grad = layer.FeedBackward(fbIn);
    auto fbOut = out_grad.Get<LayerIO>();
    static_assert(is_same<decltype(fbOut), NullParameter>::value, "Test error");

    GradCollector<float, DeviceTags::CPU> grad_collector;
    layer.GradCollect(grad_collector);
    assert(grad_collector.size() == 2);

    bool weight_update_valid = false;
    bool bias_update_valid = false;

    for (auto& p : grad_collector)
    {
        auto w = p.weight;
        auto info = Evaluate(Collapse(p.grad));
        if (w == w1)
        {
            weight_update_valid = true;
            assert(fabs(info(0, 0) - 0.0414 * 0.1) < 0.00001);
            assert(fabs(info(0, 1) - 0.0414 * 0.2) < 0.00001);
            assert(fabs(info(1, 0) - 0.04706511 * 0.1) < 0.00001);
            assert(fabs(info(1, 1) - 0.04706511 * 0.2) < 0.00001);
            assert(fabs(info(2, 0) + 0.02852585 * 0.1) < 0.00001);
            assert(fabs(info(2, 1) + 0.02852585 * 0.2) < 0.00001);
        }
        else if (w == b1)
        {
            bias_update_valid = true;
            assert(fabs(info(0, 0) - 0.0414) < 0.00001);
            assert(fabs(info(1, 0) - 0.04706511) < 0.00001);
            assert(fabs(info(2, 0) + 0.02852585) < 0.00001);
        }
        else
        {
            assert(false);
        }
    }
    assert(bias_update_valid);
    assert(weight_update_valid);

    params.clear();
    layer.SaveWeights(params);
    assert(params.find("root-weight") != params.end());
    assert(params.find("root-bias") != params.end());

    cout << "done" << endl;
}
}

void test_single_layer()
{
    test_single_layer1();
    test_single_layer2();
    test_single_layer3();
    test_single_layer4();
}
