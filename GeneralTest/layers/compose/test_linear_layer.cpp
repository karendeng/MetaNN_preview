#include <MetaNN/meta_nn.h>
#include "../../facilities/data_gen.h"
#include <cassert>
#include <map>
#include <iostream>
using namespace MetaNN;
using namespace std;

namespace
{
void test_linear_layer1()
{
    cout << "Test linear layer case 1 ...\t";
    using RootLayer = MakeLayer<LinearLayer>;
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

    assert(fabs(out(0, 0) - 0.75f) < 0.00001);
    assert(fabs(out(1, 0) - 0.91f) < 0.00001);
    assert(fabs(out(2, 0) - 1.07f) < 0.00001);

    auto out_grad = layer.FeedBackward(LayerIO::Create());
    auto fbOut = out_grad.Get<LayerIO>();
    static_assert(is_same<decltype(fbOut), NullParameter>::value, "Test error");

    params.clear();
    layer.SaveWeights(params);
    assert(params.find("root-weight") != params.end());
    assert(params.find("root-bias") != params.end());

    cout << "done" << endl;
}

void test_linear_layer2()
{
    cout << "Test linear layer case 2 ...\t";
    using RootLayer = MakeLayer<LinearLayer, PUpdate>;
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

    assert(fabs(out(0, 0) - 0.75f) < 0.00001);
    assert(fabs(out(1, 0) - 0.91f) < 0.00001);
    assert(fabs(out(2, 0) - 1.07f) < 0.00001);

    Matrix<float, DeviceTags::CPU> g(3, 1);
    g.SetValue(0, 0, 0.1f);
    g.SetValue(1, 0, 0.2f);
    g.SetValue(2, 0, 0.3f);
    auto fbIn = LayerIO::Create().Set<LayerIO>(g);
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

            auto tmp = Evaluate(Dot(g, Transpose(i)));
            assert(tmp.RowNum() == info.RowNum());
            assert(tmp.ColNum() == info.ColNum());

            for (size_t i = 0; i < tmp.RowNum(); ++i)
            {
                for (size_t j = 0; j < tmp.ColNum(); ++j)
                {
                    assert(fabs(info(i, j) - tmp(i, j)) < 0.0001f);
                }
            }
        }
        else if (w == b1)
        {
            bias_update_valid = true;
            for (size_t i = 0; i < info.RowNum(); ++i)
            {
                for (size_t j = 0; j < info.ColNum(); ++j)
                {
                    assert(fabs(info(i, j) - g(i, j)) < 0.0001f);
                }
            }
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

void test_linear_layer3()
{
    cout << "Test linear layer case 3 ...\t";
    using RootLayer = MakeLayer<LinearLayer, PUpdate,
                                             SubPolicyContainer<Sublayerof<LinearLayer>::Weight, PNoUpdate>>;
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

    assert(fabs(out(0, 0) - 0.75f) < 0.00001);
    assert(fabs(out(1, 0) - 0.91f) < 0.00001);
    assert(fabs(out(2, 0) - 1.07f) < 0.00001);

    Matrix<float, DeviceTags::CPU> g(3, 1);
    g.SetValue(0, 0, 0.1f);
    g.SetValue(1, 0, 0.2f);
    g.SetValue(2, 0, 0.3f);
    auto fbIn = LayerIO::Create().Set<LayerIO>(g);
    auto out_grad = layer.FeedBackward(fbIn);
    auto fbOut = out_grad.Get<LayerIO>();
    static_assert(is_same<decltype(fbOut), NullParameter>::value, "Test error");

    GradCollector<float, DeviceTags::CPU> grad_collector;
    layer.GradCollect(grad_collector);
    assert(grad_collector.size() == 1);

    auto w = (*grad_collector.begin()).weight;
    auto info = Evaluate(Collapse((*grad_collector.begin()).grad));
    assert(w == b1);
    for (size_t i = 0; i < info.RowNum(); ++i)
    {
        for (size_t j = 0; j < info.ColNum(); ++j)
        {
            assert(fabs(info(i, j) - g(i, j)) < 0.0001f);
        }
    }

    params.clear();
    layer.SaveWeights(params);
    assert(params.find("root-weight") != params.end());
    assert(params.find("root-bias") != params.end());
    cout << "done" << endl;
}

void test_linear_layer4()
{
    cout << "Test linear layer case 4 ...\t";
    using RootLayer = MakeLayer<LinearLayer, PUpdate,
                                             SubPolicyContainer<Sublayerof<LinearLayer>::Bias, PNoUpdate>>;
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

    assert(fabs(out(0, 0) - 0.75f) < 0.00001);
    assert(fabs(out(1, 0) - 0.91f) < 0.00001);
    assert(fabs(out(2, 0) - 1.07f) < 0.00001);

    Matrix<float, DeviceTags::CPU> g(3, 1);
    g.SetValue(0, 0, 0.1f);
    g.SetValue(1, 0, 0.2f);
    g.SetValue(2, 0, 0.3f);
    auto fbIn = LayerIO::Create().Set<LayerIO>(g);
    auto out_grad = layer.FeedBackward(fbIn);
    auto fbOut = out_grad.Get<LayerIO>();
    static_assert(is_same<decltype(fbOut), NullParameter>::value, "Test error");

    GradCollector<float, DeviceTags::CPU> grad_collector;
    layer.GradCollect(grad_collector);
    assert(grad_collector.size() == 1);

    auto w = (*grad_collector.begin()).weight;
    assert(w == w1);

    auto check = Collapse((*grad_collector.begin()).grad);
    auto check2 = Dot(g, Transpose(i));

    auto handle1 = check.EvalRegister();
    auto handle2 = check2.EvalRegister();
    EvalPlan<DeviceTags::CPU>::Eval();

    auto info = handle1.Data();
    auto tmp = handle2.Data();
    assert(tmp.RowNum() == info.RowNum());
    assert(tmp.ColNum() == info.ColNum());

    for (size_t i = 0; i < tmp.RowNum(); ++i)
    {
        for (size_t j = 0; j < tmp.ColNum(); ++j)
        {
            assert(fabs(info(i, j) - tmp(i, j)) < 0.0001f);
        }
    }

    params.clear();
    layer.SaveWeights(params);
    assert(params.find("root-weight") != params.end());
    assert(params.find("root-bias") != params.end());
    cout << "done" << endl;
}

void test_linear_layer5()
{
    cout << "Test linear layer case 5 ...\t";
    using RootLayer = MakeLayer<LinearLayer,
                                SubPolicyContainer<Sublayerof<LinearLayer>::Bias, PUpdate>>;
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

    assert(fabs(out(0, 0) - 0.75f) < 0.00001);
    assert(fabs(out(1, 0) - 0.91f) < 0.00001);
    assert(fabs(out(2, 0) - 1.07f) < 0.00001);

    Matrix<float, DeviceTags::CPU> g(3, 1);
    g.SetValue(0, 0, 0.1f);
    g.SetValue(1, 0, 0.2f);
    g.SetValue(2, 0, 0.3f);
    auto fbIn = LayerIO::Create().Set<LayerIO>(g);
    auto out_grad = layer.FeedBackward(fbIn);
    auto fbOut = out_grad.Get<LayerIO>();
    static_assert(is_same<decltype(fbOut), NullParameter>::value, "Test error");

    GradCollector<float, DeviceTags::CPU> grad_collector;
    layer.GradCollect(grad_collector);
    assert(grad_collector.size() == 1);

    auto w = (*grad_collector.begin()).weight;
    auto info = Evaluate(Collapse((*grad_collector.begin()).grad));
    assert(w == b1);
    for (size_t i = 0; i < info.RowNum(); ++i)
    {
        for (size_t j = 0; j < info.ColNum(); ++j)
        {
            assert(fabs(info(i, j) - g(i, j)) < 0.0001f);
        }
    }

    params.clear();
    layer.SaveWeights(params);
    assert(params.find("root-weight") != params.end());
    assert(params.find("root-bias") != params.end());
    cout << "done" << endl;
}

void test_linear_layer6()
{
    cout << "Test linear layer case 6 ...\t";
    using RootLayer = MakeLayer<LinearLayer,
                                SubPolicyContainer<Sublayerof<LinearLayer>::Weight, PUpdate>>;
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

    assert(fabs(out(0, 0) - 0.75f) < 0.00001);
    assert(fabs(out(1, 0) - 0.91f) < 0.00001);
    assert(fabs(out(2, 0) - 1.07f) < 0.00001);

    Matrix<float, DeviceTags::CPU> g(3, 1);
    g.SetValue(0, 0, 0.1f);
    g.SetValue(1, 0, 0.2f);
    g.SetValue(2, 0, 0.3f);
    auto fbIn = LayerIO::Create().Set<LayerIO>(g);
    auto out_grad = layer.FeedBackward(fbIn);
    auto fbOut = out_grad.Get<LayerIO>();
    static_assert(is_same<decltype(fbOut), NullParameter>::value, "Test error");

    GradCollector<float, DeviceTags::CPU> grad_collector;
    layer.GradCollect(grad_collector);
    assert(grad_collector.size() == 1);

    auto w = (*grad_collector.begin()).weight;
    assert(w == w1);
    auto check1 = Collapse((*grad_collector.begin()).grad);
    auto check2 = Dot(g, Transpose(i));

    auto handle1 = check1.EvalRegister();
    auto handle2 = check2.EvalRegister();
    EvalPlan<DeviceTags::CPU>::Eval();

    auto info = handle1.Data();
    auto tmp = handle2.Data();

    assert(tmp.RowNum() == info.RowNum());
    assert(tmp.ColNum() == info.ColNum());

    for (size_t i = 0; i < tmp.RowNum(); ++i)
    {
        for (size_t j = 0; j < tmp.ColNum(); ++j)
        {
            assert(fabs(info(i, j) - tmp(i, j)) < 0.0001f);
        }
    }

    params.clear();
    layer.SaveWeights(params);
    assert(params.find("root-weight") != params.end());
    assert(params.find("root-bias") != params.end());
    cout << "done" << endl;
}
}

void test_linear_layer()
{
    test_linear_layer1();
    test_linear_layer2();
    test_linear_layer3();   // bias is updated, weight is not --- update default
    test_linear_layer4();   // weight is updated, bias is not --- update default
    test_linear_layer5();   // bias is updated, weight is not --- non-update default
    test_linear_layer6();   // weight is updated, bias is not --- non-update default
}
