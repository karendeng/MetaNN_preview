#pragma once

#include <MetaNN/facilities/named_params.h>
#include <MetaNN/layers/facilities/common_io.h>
#include <MetaNN/layers/facilities/policies.h>
#include <MetaNN/policies/policy_operations.h>
#include <stdexcept>

namespace MetaNN
{
using ElementMulLayerInput = NamedParams<struct ElementMulLayerIn1,
                                         struct ElementMulLayerIn2>;

namespace NSElementMulLayer
{
template <bool bFeedbackOut>
struct Process_
{
    template <typename ElementType, typename DeviceType>
    using DataType = LayerTraits::LayerInternalBufType<ElementType, DeviceType,
                                                       CategoryTags::Matrix>;

    template <typename TIn1, typename TIn2, typename InterType>
    static void Record(const TIn1& in1, const TIn2& in2,
                       InterType& data1, InterType& data2)
    {
        data1.push(MakeDynamic(in1));
        data2.push(MakeDynamic(in2));
    }

    template <typename TGrad, typename InterType>
    static auto FeedBackward(const TGrad& grad, InterType& data1, InterType& data2)
    {
        if ((data1.empty()) || (data2.empty()))
        {
            throw std::runtime_error("Cannot do FeedBackward for ElementMulLayer.");
        }

        auto top1 = data1.top();
        auto top2 = data2.top();
        data1.pop();
        data2.pop();

        auto grad_eval = grad.template Get<LayerIO>();

        return ElementMulLayerInput::Create()
                        .template Set<ElementMulLayerIn1>(grad_eval * top2)
                        .template Set<ElementMulLayerIn2>(grad_eval * top1);
    }
};

template <>
struct Process_<false>
{
    template <typename ElementType, typename DeviceType>
    using DataType = NullParameter;

    template <typename TIn1, typename TIn2, typename InterType>
    static void Record(const TIn1&, const TIn2&, InterType&, InterType&)
    {}

    template <typename TGrad, typename InterType>
    static auto FeedBackward(const TGrad&, InterType&, InterType&)
    {
        return ElementMulLayerInput::Create();
    }
};
}

template <typename TPolicies>
class ElementMulLayer
{
    static_assert(IsPolicyContainer<TPolicies>, "TPolicies is not a policy container.");
    using CurLayerPolicy = PlainPolicy<TPolicies>;
public:
    static constexpr bool IsFeedbackOutput = PolicySelect<FeedbackPolicy, CurLayerPolicy>::IsFeedbackOutput;
    static constexpr bool IsUpdate = false;
    using InputType = ElementMulLayerInput;
    using OutputType = LayerIO;

private:
    using ElementType = typename PolicySelect<OperandPolicy, CurLayerPolicy>::ElementType;
    using DeviceType = typename PolicySelect<OperandPolicy, CurLayerPolicy>::DeviceType;

    using Process_ = NSElementMulLayer::Process_<IsFeedbackOutput>;

public:
    template <typename TIn>
    auto FeedForward(const TIn& p_in)
    {
        const auto& val1 = p_in.template Get<ElementMulLayerIn1>();
        const auto& val2 = p_in.template Get<ElementMulLayerIn2>();

        using rawType1 = std::decay_t<decltype(val1)>;
        using rawType2 = std::decay_t<decltype(val2)>;

        static_assert(!std::is_same<rawType1, NullParameter>::value, "parameter1 is invalid");
        static_assert(!std::is_same<rawType2, NullParameter>::value, "parameter2 is invalid");

        Process_::Record(val1, val2, m_data1, m_data2);
        return LayerIO::Create().template Set<LayerIO>(val1 * val2);
    }

    template <typename TGrad>
    auto FeedBackward(const TGrad& p_grad)
    {
        return Process_::FeedBackward(p_grad, m_data1, m_data2);
    }

    void NeutralInvariant()
    {
        LayerTraits::NeutralInvariant(m_data1);
        LayerTraits::NeutralInvariant(m_data2);
    }

private:
    using DataType = typename Process_::template DataType<ElementType,
                                                          DeviceType>;
    DataType m_data1;
    DataType m_data2;
};
}
